"""OU & AR(1) utilities for Stat-Arb V6.

Responsibilities
----------------
- Z-score utilities for spread signals.
- AR(1) fitting on spreads -> map to continuous-time OU parameters.
- Half-life helpers used by gating/backtest.

References
------------------
Discrete AR(1): X_{t+1} = c + phi * X_t + eps_t,  eps_t ~ N(0, sigma_eps^2)
OU(kappa, mu, sigma): dX_t = kappa * (mu - X_t) dt + sigma dW_t
Mapping with step Delta t (days):
    phi   = exp(-kappa * Delta t)
    kappa = -ln(phi) / Delta t
    mu    = c / (1 - phi)
    sigma^2 = sigma_eps^2 * 2*kappa / (1 - phi^2)
Half-life (in Delta t units): HL = ln(2) / kappa. For daily data, HL is in trading days.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception as e:  # pragma: no cover - environment setup issue
    raise RuntimeError(
        "statsmodels is required (ou.py). Add it to pyproject.toml deps."
    ) from e

# -----------------------------------------------------------------------------
# Z-scores
# -----------------------------------------------------------------------------

def zscore(x: pd.Series) -> pd.Series:
    """Standard z-score: (x - mean) / std, ignoring NaNs.

    For stationary spreads, this is the canonical trading signal.
    """
    m = x.mean(skipna=True)
    s = x.std(ddof=1, skipna=True)
    if not np.isfinite(s) or s == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - m) / s


def rolling_zscore(x: pd.Series, window: int, *, min_periods: Optional[int] = None) -> pd.Series:
    """Rolling z-score with a fixed window (formation-aligned in backtests).

    Parameters
    ----------
    x : pd.Series
        The spread time series.
    window : int
        Rolling window length in observations.
    min_periods : int, optional
        Minimum observations to compute a value. Defaults to ``window``.
    """
    if min_periods is None:
        min_periods = window
    roll = x.rolling(window=window, min_periods=min_periods)
    mu = roll.mean()
    sd = roll.std(ddof=1)
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


# -----------------------------------------------------------------------------
# AR(1) fitting and OU mapping
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AR1Result:
    c: float
    phi: float
    sigma_eps: float
    n: int
    se_phi: float
    hac_se_phi: Optional[float]

    def half_life_days(self, dt: float = 1.0) -> float:
        return half_life_from_phi(self.phi, dt=dt)

    def to_ou(self, dt: float = 1.0) -> Tuple[float, float, float]:
        """Return (kappa, mu, sigma) for the continuous-time OU."""
        return ou_from_ar1(self.c, self.phi, self.sigma_eps, dt=dt)


def fit_ar1(x: pd.Series, *, use_hac: bool = True, hac_lags: Optional[int] = None) -> AR1Result:
    """Fit AR(1) with intercept via OLS and return estimates + (HAC) s.e.
    
    We regress x_t on [1, x_{t-1}] after dropping NaNs. HAC standard
    errors use Newey-West with lags ≈ 1.5*sqrt(n) by default.
    """

    s = pd.Series(x).astype(float).dropna()
    if len(s) < 20:
        raise ValueError("need at least 20 observations to fit AR(1)")

    y = s.iloc[1:].values
    xlag = s.shift(1).iloc[1:].values
    X = sm.add_constant(xlag)

    ols = sm.OLS(y, X).fit()
    c = float(ols.params[0])
    phi = float(ols.params[1])
    # Residual std
    sigma_eps = float(np.sqrt(max(ols.scale, 0.0)))

    # OLS standard error for phi
    try:
        se_phi = float(np.sqrt(np.diag(ols.cov_params()))[1])
    except Exception:
        se_phi = float("nan")

    # HAC (Newey-West) standard error
    hac_se = None
    if use_hac:
        if hac_lags is None:
            n = len(y)
            hac_lags = max(1, int(1.5 * np.sqrt(n)))
        try:
            cov_hac = ols.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)
            hac_se = float(np.sqrt(np.diag(cov_hac.cov_params()))[1])
        except Exception:
            hac_se = None

    return AR1Result(c=c, phi=phi, sigma_eps=sigma_eps, n=len(y), se_phi=se_phi, hac_se_phi=hac_se)


def half_life_from_phi(phi: float, *, dt: float = 1.0) -> float:
    """Half-life in `dt` units given AR(1) coefficient `phi`.

    For stationary AR(1), 0 < phi < 1. If phi <= 0 or >= 1, return inf.
    """
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return float("inf")
    kappa = -np.log(phi) / float(dt)
    return float(np.log(2.0) / kappa)


def ou_from_ar1(c: float, phi: float, sigma_eps: float, *, dt: float = 1.0) -> Tuple[float, float, float]:
    """Map AR(1) parameters to OU (kappa, mu, sigma) with step ``dt``.
    Returns (kappa, mu, sigma).
    """
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return (float("nan"), float("nan"), float("nan"))
    kappa = -np.log(phi) / float(dt)
    mu = c / (1.0 - phi)
    sigma2 = (sigma_eps ** 2) * 2.0 * kappa / (1.0 - phi ** 2)
    sigma = float(np.sqrt(max(sigma2, 0.0)))
    return (float(kappa), float(mu), sigma)

def rolling_robust_z(x: pd.Series, window: int) -> pd.Series:
    """Rolling robust z-score via median/MAD.

    Uses median and median absolute deviation (MAD) over a rolling window.
    min_periods defaults to window//2 for early windows. Returns NaN where
    MAD is zero or not defined.
    """
    med = x.rolling(window, min_periods=window//2).median()
    mad = (x - med).abs().rolling(window, min_periods=window//2).median()
    z = (x - med) / (1.4826 * mad.replace(0, np.nan))
    return z

__all__ = [
    "zscore",
    "rolling_zscore",
    "AR1Result",
    "fit_ar1",
    "half_life_from_phi",
    "ou_from_ar1",
]
