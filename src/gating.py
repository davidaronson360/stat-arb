"""

Gating logic: statistical checks a spread must pass to be tradable.

Required (all):
   1) ADF on spread: p < 0.05 (reject unit root) and negative adjustment slope.
   2) AR(1) on spread: 0 < rho < 1 and rho is statistically significant (HAC s.e.).
   3) Half-life HL in [5, 60] trading days; time stop = 3 x HL (returned for use).

Also includes Benjamini–Hochberg FDR control across ADF p-values.
Consumes an already-computed spread or price panel; no I/O.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import logging

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    from ou import fit_ar1, half_life_from_phi
    from formation import SpreadSpec, Candidate, build_spread
else:
    from .ou import fit_ar1, half_life_from_phi
    from .formation import SpreadSpec, Candidate, build_spread

try:
    from statsmodels.tsa.stattools import adfuller
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "statsmodels is required (gating.py). Add it to pyproject.toml deps."
    ) from e

logger = logging.getLogger("stat_arb_v6.gating")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Configs & result containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GateConfig:
    adf_alpha: float = 0.05
    half_life_bounds: Tuple[int, int] = (5, 60)
    require_negative_adf_slope: bool = True
    require_rho_in_0_1: bool = True
    rho_signif_alpha: float = 0.05  # |rho|/se > z_{1-alpha/2}
    use_log_prices: bool = True


@dataclass(frozen=True)
class GateDiagnostics:
    adf_pvalue: float
    adf_tstat: float
    adf_lag: int
    adf_nobs: int
    adf_slope: float  # coefficient on lagged level in ADF regression (should be < 0)

    rho: float
    rho_se_hac: float
    rho_t_hac: float
    rho_se_ols: float
    rho_t_ols: float

    half_life: float
    time_stop_days: float


@dataclass(frozen=True)
class GateResult:
    spec: SpreadSpec
    passed: bool
    diag: GateDiagnostics


# -----------------------------------------------------------------------------
# Core tests
# -----------------------------------------------------------------------------

def _adf_with_slope(x: pd.Series, *, autolag: str = "AIC") -> Tuple[float, float, int, int, float]:
    """Run ADF and return (p, t, usedlag, nobs, slope_on_y_lagged).
    Works with both 7-tuple (store) and 4-tuple returns; reconstructs slope if needed.
    """
    s = pd.Series(x).astype(float).dropna()
    if len(s) < 40:
        return (np.nan, np.nan, 0, len(s), np.nan)

    # Try store=True; some builds don’t support it.
    try:
        out = adfuller(s.values, maxlag=None, regression="c", autolag=autolag, store=True)
    except TypeError:
        out = adfuller(s.values, maxlag=None, regression="c", autolag=autolag)

    # 7-tuple path with stored regression
    if len(out) >= 7:
        tstat, pval, usedlag, nobs, _crit, _icbest, store = out
        slope = np.nan
        try:
            params = store.resols.params
            if hasattr(params, "index") and "y.L1" in getattr(params, "index"):
                slope = float(params.loc["y.L1"])  # type: ignore[attr-defined]
            else:
                slope = float(params[1])  # [const, y_{t-1}, dY_{t-1}, ...]
        except Exception:
            slope = np.nan
        return (float(pval), float(tstat), int(usedlag), int(nobs), float(slope))

    # 4-tuple path: (tstat, pval, usedlag, nobs)
    tstat, pval, usedlag_raw, nobs_raw = out[:4]
    # Safely coerce usedlag/nobs
    try:
        k = int(usedlag_raw)
    except Exception:
        k = 0
    try:
        nobs = int(nobs_raw)
    except Exception:
        nobs = len(s) - 1

    # Rebuild: Dy_t = a + r * y_{t-1} + sum gamma_i * Dy_{t-i} + e  and extract r
    y = s.values.astype(float)
    dy = np.diff(y)
    y_lag = y[:-1]
    if k > 0 and len(dy) > k:
        # columns: Dy_{t-1}..Dy_{t-k}
        dy_lags = np.column_stack([dy[(k - j - 1):-(j + 1) or None] for j in range(k)])
        y_reg = dy[k:]
        ylag_reg = y_lag[k:]
        X = np.column_stack([np.ones(len(ylag_reg)), ylag_reg, dy_lags])
    else:
        y_reg = dy
        ylag_reg = y_lag
        X = np.column_stack([np.ones(len(ylag_reg)), ylag_reg])

    try:
        import statsmodels.api as sm  # local import
        ols = sm.OLS(y_reg, X).fit()
        slope = float(ols.params[1])
    except Exception:
        slope = np.nan

    return (float(pval), float(tstat), int(k), int(nobs), float(slope))


def _rho_significant(phi: float, se: float, alpha: float) -> bool:
    """Two-sided z-test for rho!=0. Uses fixed z_crit≈1.95996 for alpha=0.05."""
    if not np.isfinite(phi) or not np.isfinite(se) or se <= 0:
        return False
    # If someone set a weird alpha, just clamp to 0.05 behavior
    z = abs(phi) / se
    zcrit = 1.959963984540054  # ≈ z_{0.975}
    return bool(z > zcrit)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def gate_spread(spread: pd.Series, spec: SpreadSpec, cfg: GateConfig) -> GateResult:
    """Evaluate a single spread time series against gating rules.

    Parameters
    ----------
    spread : Series
        The (log-price) spread over the *formation* window.
    spec : SpreadSpec
        Identifies the construction (for logging/reporting only).
    cfg : GateConfig
        Thresholds.
    """
    s = pd.Series(spread).astype(float).dropna()
    pval, tstat, lag, nobs, slope = _adf_with_slope(s)
    ar = fit_ar1(s, use_hac=True)

    hl = half_life_from_phi(ar.phi)
    tstop = float(3.0 * hl) if np.isfinite(hl) else float("inf")

    # Checks
    cond_adf = (np.isfinite(pval) and pval < cfg.adf_alpha)
    if cfg.require_negative_adf_slope:
        cond_adf = cond_adf and (np.isfinite(slope) and slope < 0)

    cond_rho = True
    if cfg.require_rho_in_0_1:
        cond_rho = 0.0 < ar.phi < 1.0
    cond_rho = cond_rho and _rho_significant(ar.phi, ar.hac_se_phi or np.nan, cfg.rho_signif_alpha)

    lo, hi = cfg.half_life_bounds
    cond_hl = np.isfinite(hl) and (lo <= hl <= hi)

    passed = bool(cond_adf and cond_rho and cond_hl)

    diag = GateDiagnostics(
        adf_pvalue=float(pval) if np.isfinite(pval) else float("nan"),
        adf_tstat=float(tstat) if np.isfinite(tstat) else float("nan"),
        adf_lag=int(lag),
        adf_nobs=int(nobs),
        adf_slope=float(slope) if np.isfinite(slope) else float("nan"),
        rho=float(ar.phi),
        rho_se_hac=float(ar.hac_se_phi) if ar.hac_se_phi is not None else float("nan"),
        rho_t_hac=(float(ar.phi) / float(ar.hac_se_phi) if ar.hac_se_phi else float("nan")),
        rho_se_ols=float(ar.se_phi),
        rho_t_ols=(float(ar.phi) / float(ar.se_phi) if np.isfinite(ar.se_phi) and ar.se_phi > 0 else float("nan")),
        half_life=float(hl) if np.isfinite(hl) else float("inf"),
        time_stop_days=float(tstop),
    )

    return GateResult(spec=spec, passed=passed, diag=diag)


def evaluate_candidates(
    prices: pd.DataFrame,
    candidates: Sequence[Candidate],
    cfg: GateConfig,
) -> List[GateResult]:
    """Build spreads from prices and evaluate gates for each candidate.

    `prices` should already be the *formation window* (e.g., last 252d) and in
    the same (log vs level) space used by `formation` to derive specs. In our
    pipeline we pass log prices to `formation`, so do the same here.
    """
    if cfg.use_log_prices:
        P = np.log(prices.clip(lower=1e-6))
    else:
        P = prices

    results: List[GateResult] = []
    for cand in candidates:
        cols = list(cand.spec.tickers)
        if not set(cols).issubset(P.columns):
            continue
        s = build_spread(P[cols], cand.spec)
        res = gate_spread(s, cand.spec, cfg)
        results.append(res)
    return results


# -----------------------------------------------------------------------------
# Benjamini–Hochberg FDR (p-values -> q-values, keep if q <= alpha)
# -----------------------------------------------------------------------------

def fdr_bh(pvalues: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg procedure. Returns (keep_mask, qvalues).

    Non-finite p-values are treated as 1.0.
    """
    p = np.asarray([pv if np.isfinite(pv) else 1.0 for pv in pvalues], dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    q = np.empty_like(p)
    # Compute BH adjusted q-values in sorted order, then enforce monotonicity
    q_sorted = (p[order] * n) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q[order] = np.minimum(q_sorted, 1.0)
    keep = q <= alpha
    return keep, q


__all__ = [
    "GateConfig",
    "GateDiagnostics",
    "GateResult",
    "gate_spread",
    "evaluate_candidates",
    "fdr_bh",
]
