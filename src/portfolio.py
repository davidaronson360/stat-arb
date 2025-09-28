"""Portfolio sizing and volatility targeting for Stat-Arb V6.

What this does
--------------
- Convert a set of active spreads into per-leg share quantities.
- Target a portfolio-level annualized volatility using a simple risk-parity rule.
- Enforce caps: max active spreads and max gross leverage.

Notes
-----
- Risk unit: we size on absolute spread changes (Delta S_t), not percent returns.
  For a stationary spread S_t built from (log) prices, use sigma(Delta S_t).
- Risk-parity: w_i ∝ 1/sigma_i. With N independent spreads and daily target
  sigma_port, choose k so that Var(sum_i w_i * Delta S_i) = sigma_port^2 * capital^2.
  With w_i = k / sigma_i, Var ≈ k^2 * N, so k = sigma_port * capital / sqrt(N).
- Dollar-neutral legs: we allocate each spread notional across legs in proportion
  to |weights| at current prices, apply the weight signs, and compute shares.
- Caps: after initial sizing, we scale down if gross exposure exceeds
  max_gross_leverage * capital.

API
---
- allocate(...): main entry point returning per-spread legs and achieved stats.
- PortfolioConfig: target_vol_annual, max_gross_leverage, max_active_spreads, sigma_window.

This module is intentionally small and dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import logging

import numpy as np
import pandas as pd

# --- dual-mode imports: package or script ---
if __package__ in (None, ""):
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    from formation import SpreadSpec
    from costs import Leg
else:
    from .formation import SpreadSpec
    from .costs import Leg
# --- end dual-mode imports ---


logger = logging.getLogger("stat_arb_v6.portfolio")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Configs & containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioConfig:
    target_vol_annual: float = 0.10
    max_gross_leverage: float = 2.0   # × capital
    max_active_spreads: int = 10
    sigma_window: int = 63            # days for Δspread std


@dataclass(frozen=True)
class SpreadSizing:
    spec: SpreadSpec
    sigma_daily: float
    notional: float                 # signed $ notional for the spread PnL
    legs: Tuple[Leg, ...]           # sized legs at current prices


@dataclass(frozen=True)
class AllocationResult:
    sizings: Tuple[SpreadSizing, ...]
    achieved_portfolio_vol_annual: float
    gross_exposure: float
    scale_vol_factor: float
    scale_gross_factor: float


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def allocate(
    *,
    spread_series: Mapping[SpreadSpec, pd.Series],  # level spread over formation window
    last_prices: pd.Series,  # current prices for all tickers involved (float)
    capital: float,
    cfg: PortfolioConfig,
) -> AllocationResult:
    """Risk‑parity sizing of active spreads with portfolio vol targeting.

    Parameters
    ----------
    spread_series : mapping of SpreadSpec → pd.Series
        Each series should be the *level* spread (e.g., from formation.build_spread)
        over a recent window to estimate Δspread sigma.
    last_prices : pd.Series
        Latest prices indexed by ticker; used to convert notionals to shares.
    capital : float
        Portfolio capital (for leverage scaling).
    cfg : PortfolioConfig
        Target vol and caps.

    Returns
    -------
    AllocationResult
        Per‑spread leg sizes and achieved portfolio stats.
    """
    # 1) Select up to max_active_spreads (assume upstream has already ranked)
    items = list(spread_series.items())[: cfg.max_active_spreads]
    if not items:
        return AllocationResult(sizings=tuple(), achieved_portfolio_vol_annual=0.0, gross_exposure=0.0, scale_vol_factor=1.0, scale_gross_factor=1.0)

    # 2) Compute daily sigma of Δspread
    sigmas = []
    for spec, s in items:
        sigma = _delta_sigma(s, window=cfg.sigma_window)
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = np.nan
        sigmas.append(sigma)
    sigmas = np.asarray(sigmas, dtype=float)

    valid_mask = np.isfinite(sigmas) & (sigmas > 0)
    if not valid_mask.any():
        logger.warning("All candidate spreads have invalid sigma; nothing to allocate")
        return AllocationResult(sizings=tuple(), achieved_portfolio_vol_annual=0.0, gross_exposure=0.0, scale_vol_factor=1.0, scale_gross_factor=1.0)

    # Keep only valid
    items = [itm for itm, ok in zip(items, valid_mask) if ok]
    sigmas = sigmas[valid_mask]

    N = len(items)
    target_daily_vol = _annual_to_daily_vol(cfg.target_vol_annual)

    # 3) Risk‑parity base notionals: w_i = k / sigma_i, with k set to hit target daily vol
    k = (target_daily_vol * capital) / max(np.sqrt(N), 1e-12)
    w = k / sigmas  # signed notionals per unit spread (sign handled by signal elsewhere)

    # 4) Build initial legs and compute gross exposure
    sizings: List[SpreadSizing] = []
    gross = 0.0
    for (spec, s), notional in zip(items, w):
        legs = _legs_from_notional(spec, notional, last_prices)
        gross += sum(abs(l.shares) * l.price for l in legs)
        sizings.append(SpreadSizing(spec=spec, sigma_daily=float(sigmas[len(sizings)]), notional=float(notional), legs=tuple(legs)))

    # 5) Scale for gross leverage cap
    max_gross_dollars = cfg.max_gross_leverage * capital
    scale_gross = 1.0 if gross <= max_gross_dollars or gross == 0 else (max_gross_dollars / gross)
    if scale_gross < 1.0 and scale_gross > 0:
        sizings = [
            _scale_sizing(sz, scale_gross) for sz in sizings
        ]
        gross *= scale_gross

    # 6) Estimate achieved portfolio vol (independence assumption)
    #    Var ≈ Σ ( (scale*notional)^2 * sigma_i^2 )
    scale_vol = 1.0  # we've already scaled by vol target in step 3
    port_var = 0.0
    for sz in sizings:
        port_var += (sz.notional ** 2) * (sz.sigma_daily ** 2)
    achieved_daily_vol = np.sqrt(max(port_var, 0.0)) / max(capital, 1e-12)
    achieved_annual_vol = _daily_to_annual_vol(float(achieved_daily_vol))

    return AllocationResult(
        sizings=tuple(sizings),
        achieved_portfolio_vol_annual=float(achieved_annual_vol),
        gross_exposure=float(gross),
        scale_vol_factor=float(scale_vol),
        scale_gross_factor=float(scale_gross),
    )


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _delta_sigma(s: pd.Series, *, window: int) -> float:
    s = pd.Series(s).astype(float)
    ds = s.diff().dropna()
    if window is not None and window > 0 and len(ds) >= window:
        ds = ds.tail(window)
    return float(ds.std(ddof=1)) if len(ds) > 1 else float("nan")


def _annual_to_daily_vol(vol_annual: float, *, trading_days: int = 252) -> float:
    return float(vol_annual / np.sqrt(trading_days))


def _daily_to_annual_vol(vol_daily: float, *, trading_days: int = 252) -> float:
    return float(vol_daily * np.sqrt(trading_days))


def _legs_from_notional(spec: SpreadSpec, notional: float, last_prices: pd.Series) -> List[Leg]:
    """Convert a spread notional into per‑leg dollar exposures and shares.

    We split the absolute notional across legs proportional to |weights|, then
    apply the sign of each weight. This keeps the spread dollar‑neutral.
    """
    tickers = list(spec.tickers)
    w = np.asarray(spec.weights, dtype=float)
    absw = np.abs(w)
    denom = float(absw.sum()) if absw.sum() != 0 else 1.0
    leg_notional = (absw / denom) * abs(float(notional))
    # Apply sign of weights and the sign of notional
    signed_leg_notional = np.sign(w) * np.sign(notional) * leg_notional

    legs: List[Leg] = []
    for tk, dollars in zip(tickers, signed_leg_notional):
        px = float(last_prices.get(tk, np.nan))
        if not np.isfinite(px) or px <= 0:
            raise ValueError(f"Missing/invalid price for {tk} in _legs_from_notional")
        shares = float(dollars / px)
        legs.append(Leg(ticker=tk, shares=shares, price=px))
    return legs


def _scale_sizing(sz: SpreadSizing, factor: float) -> SpreadSizing:
    legs = tuple(Leg(ticker=l.ticker, shares=l.shares * factor, price=l.price) for l in sz.legs)
    return SpreadSizing(spec=sz.spec, sigma_daily=sz.sigma_daily, notional=sz.notional * factor, legs=legs)


__all__ = [
    "PortfolioConfig",
    "SpreadSizing",
    "AllocationResult",
    "allocate",
]
