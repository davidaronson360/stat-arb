"""Formation logic for cointegrated spreads.

Scans within-industry assets and proposes stationary spread specifications
(pairs via Engle-Granger, baskets via Johansen). Operates on log prices and
returns immutable SpreadSpec objects with diagnostics for later gating and
backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple
from itertools import combinations
import logging

import numpy as np
import pandas as pd

try:  # Light, targeted imports from statsmodels
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint as eg_coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
except Exception as e:  # pragma: no cover - environment setup issue is actionable
    raise RuntimeError(
        "statsmodels is required (formation.py). Add it to pyproject.toml deps."
    ) from e

logger = logging.getLogger("stat_arb_v6.formation")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Core data structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SpreadSpec:
    """A stationary spread definition: weights over tickers + optional intercept.

    The spread at time *t* is ``intercept + sum_i weights[i] * P_i(t)`` where
    ``P_i`` are (log) prices for ``tickers[i]``.

    We keep the spec minimal and immutable so it can be serialized and reused
    across formation and trading windows.
    """

    tickers: Tuple[str, ...]
    weights: Tuple[float, ...]  # aligned to tickers
    intercept: float
    method: str  # "EG" | "Johansen"
    industry: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "industry": self.industry,
            "tickers": list(self.tickers),
            "weights": list(self.weights),
            "intercept": float(self.intercept),
        }


@dataclass(frozen=True)
class Candidate:
    """A proposed cointegrated spread plus discovery diagnostics.

    Diagnostics are intentionally lightweight; rigorous gates live in
    ``gating.py``. We include fields useful for ranking and for later auditing.
    """

    spec: SpreadSpec
    # A single scalar for coarse ranking (lower is better when comparable).
    score: float
    # Method-specific details for transparency
    details: Dict[str, float]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def discover_candidates(
    prices: pd.DataFrame,
    industries: Mapping[str, Sequence[str]],
    *,
    use_log_prices: bool = True,
    johansen_ks: Sequence[int] = (3, 4, 5),
    johansen_det_order: int = 0,
    johansen_k_ar_diff: int = 1,
) -> List[Candidate]:
    """Scan within-industry combinations and return candidate stationary spreads.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Wide matrix of prices indexed by date, columns = tickers.
    industries : Mapping[str, Sequence[str]]
        Industry -> tickers mapping.
    use_log_prices : bool, default True
        Apply log() to prices prior to tests/weights.
    
    Returns
    -------
    list[Candidate]
        One Candidate per EG pair or per significant Johansen eigenvector.
    """
    if prices.empty:
        return []

    P = np.log(prices.clip(lower=1e-6)) if use_log_prices else prices.copy()
    P = P.dropna(how="all", axis=0).dropna(how="all", axis=1)

    cands: List[Candidate] = []

    for industry, tickers in industries.items():
        cols = [t for t in tickers if t in P.columns]
        if len(cols) < 2:
            logger.info("Industry %s has <2 tickers with data; skipping", industry)
            continue
        frame = P[cols].dropna(how="any")  # strict: require all series present
        if len(frame) < 60:  # need a reasonable window length
            logger.info("Industry %s has <60 obs in formation; skipping", industry)
            continue

        # ---- EG pairs
        for a, b in combinations(cols, 2):
            try:
                cand = _eg_pair_candidate(frame[[a, b]], industry)
                if cand is not None:
                    cands.append(cand)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("EG failed for %s/%s: %s", a, b, e)

        # ---- Johansen baskets
        for k in johansen_ks:
            if len(cols) < k:
                continue
            for combo in combinations(cols, k):
                try:
                    jcands = _johansen_candidates(
                        frame[list(combo)],
                        industry=industry,
                        det_order=johansen_det_order,
                        k_ar_diff=johansen_k_ar_diff,
                    )
                    cands.extend(jcands)
                except np.linalg.LinAlgError as e:  # singular/ill-conditioned
                    logger.warning("Johansen ill-conditioned for %s: %s", combo, e)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("Johansen failed for %s: %s", combo, e)

    logger.info("Formation discovered %d candidates", len(cands))
    return cands


# -----------------------------------------------------------------------------
# Engle-Granger pairs
# -----------------------------------------------------------------------------

def _eg_pair_candidate(frame: pd.DataFrame, industry: str) -> Candidate | None:
    """Build a Candidate for a pair using EG (OLS hedge + ADF on residuals).
    
    Ranking score: EG p-value (lower is better). Returns None on failure or insufficient data.
    """
    assert frame.shape[1] == 2
    x, y = frame.columns[0], frame.columns[1]
    X = sm.add_constant(frame[x].values)
    Y = frame[y].values
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(Y)):
        return None

    ols = sm.OLS(Y, X).fit()
    a, b = float(ols.params[0]), float(ols.params[1])
    resid = Y - (a + b * frame[x].values)

    # Engle-Granger cointegration test on (Y, X); use residual-based ADF inside
    try:
        tstat, pval, _ = eg_coint(frame[y].values, frame[x].values)
    except Exception:
        return None

    spec = SpreadSpec(
        tickers=(x, y),
        weights=(-b, 1.0),  # spread = a + (-b)*X + 1*Y  (i.e., Y - b X + a)
        intercept=a,
        method="EG",
        industry=industry,
    )

    details = {
        "eg_pvalue": float(pval),
        "eg_tstat": float(tstat),
        "ols_beta": b,
        "ols_const": a,
        "resid_std": float(np.std(resid, ddof=1)),
        "n_obs": float(len(frame)),
    }
    return Candidate(spec=spec, score=float(pval), details=details)


# -----------------------------------------------------------------------------
# Johansen baskets (k>=3)
# -----------------------------------------------------------------------------

def _johansen_candidates(
    frame: pd.DataFrame,
    *,
    industry: str,
    det_order: int,
    k_ar_diff: int,
) -> List[Candidate]:
    """Return one Candidate per significant Johansen cointegration vector.
    
    Rank selection via trace statistic at 5%: if rank >= 1, emit each beta column as a spread.
    Ranking score = -margin, where margin = trace_stat - crit_95 at the selected rank
    (larger margin -> stronger relationship -> lower score).
    """

    cols = list(frame.columns)
    data = frame[cols].values
    if not np.isfinite(data).all() or len(frame) < 60:
        return []

    joh = coint_johansen(data, det_order, k_ar_diff)
    # Determine rank r via trace statistic at 5% (index 1 in cvt)
    # statsmodels returns lr1 (trace stats) and cvt (critical values 90/95/99)
    lr1 = np.asarray(getattr(joh, "lr1"))
    cvt = np.asarray(getattr(joh, "cvt"))
    if lr1.ndim != 1 or cvt.shape != (len(cols), 3):
        # Fallback safety if API changes
        return []

    r = 0
    for i in range(len(cols)):
        if lr1[i] > cvt[i, 1]:  # 95% level
            r = i + 1
    if r == 0:
        return []

    # Extract beta (cointegrating vectors). Some versions expose evec/eigvec
    beta = getattr(joh, "eigvec", None)
    if beta is None:
        beta = getattr(joh, "evec", None)
    if beta is None:
        return []
    beta = np.asarray(beta)
    # beta has shape (k, k). Take the first r columns
    beta = beta[:, :r]

    # Compute a simple strength margin at the selected rank r
    margin = float(lr1[r - 1] - cvt[r - 1, 1])

    cands: List[Candidate] = []
    for j in range(beta.shape[1]):
        vec = beta[:, j].astype(float)
        w = _normalize_vector(vec, anchor_index=len(cols) - 1)  # last ticker = -1
        spec = SpreadSpec(
            tickers=tuple(cols),
            weights=tuple(w.tolist()),
            intercept=0.0,  # we use det_order=0; intercept handled in VECM if needed
            method="Johansen",
            industry=industry,
        )
        details = {
            "j_rank": float(r),
            "j_margin95": margin,
            "vec_l2": float(np.linalg.norm(vec)),
            "n_obs": float(len(frame)),
        }
        # Lower score is better: we take -margin so stronger relationships rank first
        cands.append(Candidate(spec=spec, score=-margin, details=details))

    return cands


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def build_spread(series: pd.DataFrame, spec: SpreadSpec) -> pd.Series:
    """Construct the spread time series from (log) price columns and a spec.

    Parameters
    ----------
    series : DataFrame
        Must contain the columns in ``spec.tickers``.
    spec : SpreadSpec
        Weights/intercept definition.

    Returns
    -------
    pd.Series
        The spread (aligned to index of ``series``), name = method:tickers.
    """
    s = series[list(spec.tickers)].copy()
    arr = s.values @ np.asarray(spec.weights, dtype=float)
    arr = arr + float(spec.intercept)
    out = pd.Series(arr, index=s.index)
    out.name = f"{spec.method}:{','.join(spec.tickers)}"
    return out


def _normalize_vector(v: np.ndarray, *, anchor_index: int = -1) -> np.ndarray:
    """Normalize cointegration vector so that the anchor coefficient is ``-1``.

    This yields intuitive hedge weights relative to the anchor asset.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    a = v[anchor_index]
    if a == 0:
        # fallback: set L1 norm = 1
        s = np.sum(np.abs(v))
        return v / (s if s != 0 else 1.0)
    return -v / a

# Gating computes ADF/AR(1)/HL later
__all__ = [
    "SpreadSpec",
    "Candidate",
    "discover_candidates",
    "build_spread",
]  
