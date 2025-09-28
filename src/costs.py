"""Trading cost model for Stat‑Arb V6.

Transparent, parameterized cost components:
- **Commission**: $0.001 per share (configurable).
- **Half‑spread**: pay 1 tick per entry/exit leg (configurable), approximated as
  half of the minimum price increment. We assume a conservative fixed tick size
  of 1 cent for US equities unless provided.
- **Slippage**: a single parameter in **basis points** applied to notional.

This module is stateless: given executed trades (shares and prices), compute the
costs per leg and aggregate summaries for reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Dict
import pandas as pd

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CostConfig:
    commission_per_share: float = 0.001
    half_spread_ticks: float = 1.0
    tick_size: float = 0.01
    slippage_bps: float = 2.0


@dataclass(frozen=True)
class Leg:
    ticker: str
    shares: float  # signed (+ buy, - sell)
    price: float   # execution price

@dataclass(frozen=True)
class Trade:
    """A multi‑leg trade for a single spread at a given timestamp.

    We keep this generic: you can have N legs (e.g., k‑asset basket)."""
    when: pd.Timestamp
    legs: Tuple[Leg, ...]


@dataclass(frozen=True)
class CostBreakdown:
    commission: float
    half_spread: float
    slippage: float

    @property
    def total(self) -> float:
        return float(self.commission + self.half_spread + self.slippage)


# -----------------------------------------------------------------------------
# Core calculations
# -----------------------------------------------------------------------------

def _notional(legs: Sequence[Leg]) -> float:
    return float(sum(abs(l.shares) * l.price for l in legs))


def cost_per_trade(trade: Trade, cfg: CostConfig) -> CostBreakdown:
    """Compute commission + half‑spread + slippage for a Trade.

    Commission: sum(|shares|) * commission_per_share
    Half‑spread: sum(|shares|) * half_spread_ticks * tick_size
    Slippage: notional * (slippage_bps / 1e4)
    """
    comm = sum(abs(l.shares) for l in trade.legs) * cfg.commission_per_share
    hspr = sum(abs(l.shares) for l in trade.legs) * cfg.half_spread_ticks * cfg.tick_size
    slip = _notional(trade.legs) * (cfg.slippage_bps / 1e4)
    return CostBreakdown(commission=float(comm), half_spread=float(hspr), slippage=float(slip))


def summarize_costs(trades: Sequence[Trade], cfg: CostConfig) -> Dict[str, float]:
    """Aggregate costs across many trades → dict for quick tables.

    Returns keys: commission, half_spread, slippage, total, per_trade_avg.
    """
    if not trades:
        return {k: 0.0 for k in ["commission", "half_spread", "slippage", "total", "per_trade_avg"]}

    parts = [cost_per_trade(t, cfg) for t in trades]
    commission = float(sum(p.commission for p in parts))
    half_spread = float(sum(p.half_spread for p in parts))
    slippage = float(sum(p.slippage for p in parts))
    total = commission + half_spread + slippage
    return {
        "commission": commission,
        "half_spread": half_spread,
        "slippage": slippage,
        "total": total,
        "per_trade_avg": total / float(len(trades)),
    }


# -----------------------------------------------------------------------------
# Sensitivity helpers
# -----------------------------------------------------------------------------

def with_sensitivity(cfg: CostConfig, *, slippage_sensitivity: float = 0.5) -> Tuple[CostConfig, CostConfig]:
    """Return (low, high) configs for ±slippage sensitivity (default ±50%)."""
    low = CostConfig(
        commission_per_share=cfg.commission_per_share,
        half_spread_ticks=cfg.half_spread_ticks,
        tick_size=cfg.tick_size,
        slippage_bps=max(0.0, cfg.slippage_bps * (1.0 - slippage_sensitivity)),
    )
    high = CostConfig(
        commission_per_share=cfg.commission_per_share,
        half_spread_ticks=cfg.half_spread_ticks,
        tick_size=cfg.tick_size,
        slippage_bps=cfg.slippage_bps * (1.0 + slippage_sensitivity),
    )
    return low, high


__all__ = [
    "CostConfig",
    "Leg",
    "Trade",
    "CostBreakdown",
    "cost_per_trade",
    "summarize_costs",
    "with_sensitivity",
]
