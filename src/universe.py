"""Universe utilities for Stat‑Arb V6.

Responsibilities
----------------
- Parse `configs/base.yaml` to obtain industries, tickers, and date bounds.
- Normalize and validate the universe (uppercase, deduplicate, per‑industry count).
- Provide a `Universe` dataclass with helpers used downstream.

Design notes
------------
- We keep this module *data‑source agnostic* (no network/yfinance calls). The
  `data.py` module is responsible for fetching/caching prices. Here we only
  prepare a clean mapping of industries→tickers and the date window.
- If YAML end_date is "today", resolve it at load.
- Validation focuses on basic structural checks (duplicates, empty industries,
  min count per industry). Formation/backtest code can enforce stricter rules
  using realized data availability.

"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import logging
from pathlib import Path
from typing import Dict, List, Mapping

import yaml

logger = logging.getLogger("stat_arb_v6.universe")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize_ticker(t: str) -> str:
    return t.strip().upper()


def _parse_date_like(x: str | date | datetime) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"today", "now"}:
            return datetime.utcnow().date()
        # Accept ISO (YYYY-MM-DD)
        try:
            y, m, d = map(int, s.split("-"))
            return date(y, m, d)
        except Exception as e:  # pragma: no cover - defensive path
            raise ValueError(f"Unrecognized date format: {x!r}") from e
    raise TypeError("Expected date/datetime/str for date fields")


# -----------------------------------------------------------------------------
# Dataclass
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Universe:
    """Industry‑scoped ticker universe and date bounds.

    Attributes
    ----------
    industries : Mapping[str, tuple[str, ...]]
        Immutable mapping of industry name → tuple of tickers.
    start : date
        Inclusive start date for data.
    end : date
        Inclusive end date for data (the *request* end; downstream may use
        closed‑open intervals).
    """

    industries: Mapping[str, tuple[str, ...]]
    start: date
    end: date

    # -----------------------------
    # Construction
    # -----------------------------
    @staticmethod
    def from_yaml(path: str | Path) -> "Universe":
        """Load from a YAML file (e.g., `configs/base.yaml`).

        Expected YAML shape:
        
        universe:
          start_date: 2016-01-01
          end_date: today
          industries:
            Energy: [XOM, CVX, BP, COP, TOT]
            Airlines: [DAL, AAL, UAL, LUV, JBLU]
        """
        p = Path(path)
        cfg = yaml.safe_load(p.read_text())
        ucfg = cfg.get("universe", {})
        start = _parse_date_like(ucfg.get("start_date"))
        end = _parse_date_like(ucfg.get("end_date", "today"))
        raw_ind = ucfg.get("industries", {})
        if not raw_ind:
            raise ValueError("configs/base.yaml: universe.industries is empty")

        industries: Dict[str, tuple[str, ...]] = {}
        for name, tickers in raw_ind.items():
            if not isinstance(tickers, (list, tuple)):
                raise TypeError(f"Industry {name!r} tickers must be a list")
            norm = tuple(sorted({_normalize_ticker(t) for t in tickers if str(t).strip()}))
            if not norm:
                logger.warning("Industry %s has no tickers after normalization", name)
            industries[str(name)] = norm

        uni = Universe(industries=industries, start=start, end=end)
        # Basic validation with a friendly log, but do not raise unless fatal
        uni._log_basic_stats()
        return uni

    # -----------------------------
    # Introspection helpers
    # -----------------------------
    def by_industry(self) -> Mapping[str, tuple[str, ...]]:
        return self.industries

    def tickers(self) -> List[str]:
        """Return a sorted list of all unique tickers across industries."""
        all_t = set()
        for ts in self.industries.values():
            all_t.update(ts)
        return sorted(all_t)

    def industry_of(self, ticker: str) -> List[str]:
        """Return all industries containing `ticker` (duplicates are allowed).

        Downstream can decide how to handle multi‑membership (e.g., prefer the
        first occurrence or drop duplicates).
        """
        t = _normalize_ticker(ticker)
        return [name for name, ts in self.industries.items() if t in ts]

    # -----------------------------
    # Validation
    # -----------------------------
    def validate(self, *, min_per_industry: int = 5, allow_multi_membership: bool = True) -> None:
        """Raise if the universe violates basic structural rules.

        Parameters
        ----------
        min_per_industry : int
            Minimum number of tickers required per industry.
        allow_multi_membership : bool
            If False, raise when a ticker appears in >1 industry.
        """
        # Empty or underfilled industries
        for name, ts in self.industries.items():
            if len(ts) < min_per_industry:
                raise ValueError(
                    f"Industry {name!r} has {len(ts)} tickers; require >= {min_per_industry}"
                )

        # Duplicate membership
        if not allow_multi_membership:
            owner: Dict[str, str] = {}
            for name, ts in self.industries.items():
                for t in ts:
                    if t in owner and owner[t] != name:
                        raise ValueError(f"Ticker {t} appears in {owner[t]} and {name}")
                    owner[t] = name

        if self.end < self.start:
            raise ValueError("Universe end_date must be on/after start_date")

    # -----------------------------
    # Logging / display
    # -----------------------------
    def _log_basic_stats(self) -> None:
        ind_cnt = len(self.industries)
        tick_cnt = len(self.tickers())
        logger.info("Universe: %d industries, %d unique tickers (%s → %s)",
                    ind_cnt, tick_cnt, self.start.isoformat(), self.end.isoformat())
        for name, ts in self.industries.items():
            logger.info("  - %-28s %2d tickers", name, len(ts))


# -----------------------------------------------------------------------------
# Convenience loader for other modules
# -----------------------------------------------------------------------------

def load_universe(config_path: str | Path = "../configs/base.yaml") -> Universe:
    """Load and validate the base universe.

    Modules can use this to quickly grab the shared universe.
    """
    u = Universe.from_yaml(config_path)
    # Enforce a default minimum of 5 tickers per industry
    u.validate(min_per_industry=5, allow_multi_membership=True)
    return u


__all__ = ["Universe", "load_universe"]
