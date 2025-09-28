"""Data utilities for Stat-Arb V6.

Responsibilities
----------------
- Download Adjusted Close prices with dividends/splits applied (via yfinance).
- Deterministic local caching to small Parquet files under ``data/cache``.
- Lightweight validation & alignment to a daily (Mon-Fri) calendar.

Notes
-----
- We purposefully avoid heavyweight market‑calendar deps. We align to a
  Monday-Friday daily calendar and then prune days where **all** assets are NA.
- We cache by a stable key of (sorted tickers, start, end, interval).
- Incremental updates: if a cache exists but the requested window extends
  beyond the cached one, we fetch only the missing tail and merge.

"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as e:  # pragma: no cover - import error is actionable for user
    raise RuntimeError(
        "yfinance is required for data downloading. Add it to pyproject.toml."
    ) from e

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("stat_arb_v6.data")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class CacheKey:
    tickers: tuple[str, ...]
    start: date
    end: date
    interval: str = "1d"

    def key_str(self) -> str:
        """Stable, readable key snippet for filenames."""
        tks = ",".join(self.tickers)
        return f"{tks}__{self.start.isoformat()}__{self.end.isoformat()}__{self.interval}"

    def digest(self) -> str:
        """Short hash to avoid filesystem limits on long ticker lists."""
        h = hashlib.sha256(self.key_str().encode()).hexdigest()[:16]
        return h

    def path(self) -> Path:
        return CACHE_DIR / f"prices_{self.digest()}.parquet"


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def load_prices(
    tickers: Sequence[str],
    start: date | datetime,
    end: date | datetime,
    interval: str = "1d",
    *,
    max_forward_fill_days: int = 5,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Load Adjusted Close prices for ``tickers`` within ``[start, end]``.

    The function will honor a deterministic local cache and only hit the network
    for missing data. Outputs are a clean, aligned price matrix suitable for
    downstream formation/backtest logic.

    Parameters
    ----------
    tickers : Sequence[str]
        Equity tickers understood by Yahoo Finance.
    start, end : date | datetime
        Inclusive start, exclusive end (like pandas slicing). If you pass a
        ``date``, it is treated as midnight in the system timezone.
    interval : str, default "1d"
        yfinance interval ("1d", "1wk", "1mo"). We stick to "1d" for V6.
    max_forward_fill_days : int, default 5
        Maximum number of business days to forward‑fill gaps per series. This
        is a guardrail against illiquid listings.
    allow_partial : bool, default False
        If False, tickers with fewer than 70% observed days in the window are
        dropped. If True, they are kept after forward‑fill/back‑fill.

    Returns
    -------
    pd.DataFrame
        Indexed by naive ``DatetimeIndex`` (UTC‑naive), columns = tickers,
        float32 dtype, strictly increasing index.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty sequence")

    _start = _to_datetime(start)
    _end = _to_datetime(end)
    if _end <= _start:
        raise ValueError("end must be after start")

    tickers = tuple(sorted(set([t.strip().upper() for t in tickers if t.strip()])))
    key = CacheKey(tickers, _start.date(), (_end - timedelta(days=0)).date(), interval)

    # Try cache, then incrementally update if needed
    df = _load_cache_if_any(key)

    if df is None:
        logger.info("cache miss -> downloading %d tickers", len(tickers))
        df = _download_prices(tickers, _start, _end, interval)
        _save_cache(key, df)
    else:
        # If cached doesn't extend to requested end, fetch the tail
        cached_end = df.index.max()
        if cached_end is None or cached_end < (_end - pd.Timedelta(days=1)):
            tail_start = (cached_end + pd.Timedelta(days=1)) if cached_end is not None else _start
            logger.info("cache hit; extending tail %s -> %s", tail_start.date(), (_end - pd.Timedelta(days=1)).date())
            tail = _download_prices(tickers, tail_start, _end, interval)
            if not tail.empty:
                df = _union_prices(df, tail)
                _save_cache(key, df)

    df = _align_and_clean(df, _start, _end, max_forward_fill_days, allow_partial)
    return df


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------

def _to_datetime(x: date | datetime) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_localize(None)
    if isinstance(x, datetime):
        return pd.Timestamp(x.replace(tzinfo=None))
    if isinstance(x, date):
        return pd.Timestamp(datetime(x.year, x.month, x.day))
    raise TypeError("start/end must be date or datetime")


def _download_prices(
    tickers: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    *,
    chunk_size: int = 50,
    sleep_secs: float = 0.0,
    max_retries: int = 2,
) -> pd.DataFrame:
    """Download Adj Close matrix via yfinance (chunked, with basic retries)."""
    frames: List[pd.DataFrame] = []
    tickers_list = list(tickers)

    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i : i + chunk_size]
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    "download %d/%d: %s", i + 1, len(tickers_list), ",".join(chunk)
                )
                data = yf.download(
                    tickers=" ".join(chunk),
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval=interval,
                    auto_adjust=True,  # dividends/splits applied
                    progress=False,
                    threads=True,
                    group_by="column",
                )
                # yfinance returns a multi-index columns when multiple symbols
                # Select Adj Close across symbols, normalize into wide DataFrame
                frame = _extract_adj_close(data)
                frames.append(frame)
                break
            except Exception as err:  # pragma: no cover - network path
                last_err = err
                if attempt < max_retries:
                    logger.warning("retry %d after error: %s", attempt + 1, err)
                    if sleep_secs:
                        import time as _t

                        _t.sleep(sleep_secs)
                else:
                    raise
        if sleep_secs:
            import time as _t

            _t.sleep(sleep_secs)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out = out.loc[~out.index.duplicated()].sort_index()
    # Ensure only requested tickers present (defensive in case of delisted aliases)
    out = out.reindex(columns=list(tickers))
    return out


def _extract_adj_close(raw: pd.DataFrame) -> pd.DataFrame:
    """Handle the variety of shapes yfinance returns and get a wide Adj Close.

    Cases handled:
    - Single ticker -> columns like ['Open','High',...,'Adj Close']
    - Multi-ticker -> column MultiIndex (field, ticker)
    - Occasionally, 'Adj Close' may be missing; we fallback to 'Close'.
    """
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # Expect level 0 = field, level 1 = ticker
        if ("Adj Close" in raw.columns.get_level_values(0)):
            adj = raw["Adj Close"].copy()
        else:
            adj = raw["Close"].copy()
        adj.columns = [str(c) for c in adj.columns]
        return adj

    # Single ticker path
    cols = [c.lower().strip() for c in raw.columns]
    if "adj close" in cols:
        series = raw.iloc[:, cols.index("adj close")]
    elif "close" in cols:
        series = raw.iloc[:, cols.index("close")]
    else:  # pragma: no cover - unexpected shape
        raise ValueError("Unexpected yfinance frame shape; no Close/Adj Close found")
    series = series.rename(raw.columns[cols.index("adj close")] if "adj close" in cols else raw.columns[cols.index("close")])
    return series.to_frame()


def _align_and_clean(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_ffill_days: int,
    allow_partial: bool,
) -> pd.DataFrame:
    if df.empty:
        return df

    # Coerce dtypes & ordering
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    # Create a Mon-Fri calendar in the requested closed-open window [start, end)
    cal = pd.bdate_range(start=start, end=end - pd.Timedelta(days=1), freq="C")
    df = df.reindex(cal)

    # Drop days where **all** prices are NA (e.g., common US holiday)
    df = df.loc[~df.isna().all(axis=1)]

    # Basic quality: forward fill up to N bdays per column; then backfill day 1 only
    if max_ffill_days > 0:
        df = _bounded_ffill(df, limit=max_ffill_days)
    df = df.bfill(limit=1)

    # Optionally drop sparse tickers (e.g., illiquid/foreign listings)
    if not allow_partial:
        ok_mask = df.notna().sum(axis=0) >= int(0.7 * len(df))
        dropped = [c for c, ok in ok_mask.items() if not ok]
        if dropped:
            logger.info("dropping %d sparse tickers: %s", len(dropped), ",".join(dropped))
        df = df.loc[:, ok_mask]

    # Final coercions
    df = df.astype(np.float32)
    df = df.loc[~df.index.duplicated()].sort_index()
    return df


def _bounded_ffill(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Forward-fill with a per-column maximum span of ``limit`` business days.

    Pandas' ``ffill(limit=...)`` applies the limit to *consecutive* NaNs blocks,
    which is what we want, but we also guard against columns that start as NaN
    by applying a single-step backfill first.
    """
    f = frame.copy()
    f = f.bfill(limit=1)
    f = f.ffill(limit=limit)
    return f


def _load_cache_if_any(key: CacheKey) -> Optional[pd.DataFrame]:
    path = key.path()
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        # Safety: ensure columns are strings (tickers)
        df.columns = [str(c) for c in df.columns]
        return df
    except Exception as e:  # pragma: no cover - corrupted cache
        logger.warning("failed to read cache %s: %s; ignoring", path.name, e)
        return None


def _save_cache(key: CacheKey, df: pd.DataFrame) -> None:
    path = key.path()
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, compression="zstd", index=True)
    tmp.replace(path)


def _union_prices(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Union two price matrices by index and columns (favoring newer 'b')."""
    # Align columns union, prefer b's values when overlapping and newer index
    cols = sorted(set(a.columns) | set(b.columns))
    c = a.reindex(columns=cols).combine_first(b.reindex(columns=cols))
    # Where both have values, take 'b' for rows >= b.index.min()
    if not b.empty:
        c.loc[b.index, cols] = b.reindex(columns=cols)
    c = c.loc[~c.index.duplicated()].sort_index()
    return c


__all__ = [
    "load_prices",
    "CacheKey",
]
