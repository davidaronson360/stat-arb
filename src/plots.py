"""Plot helpers for Stat-Arb V6.

Generates:
- equity curve (cum P&L)
- rolling Sharpe (63d)
- drawdown curve

Usage
-----
From Python:
    from plots import make_all_plots
    art = make_all_plots(equity, outdir="../results/figures")

CLI (runs a fresh backtest and plots):
    python -m plots
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    if __package__ in (None, ""):
        import sys, pathlib
        sys.path.append(str(pathlib.Path(__file__).resolve().parent))
        import backtest as bt
    else:
        from . import backtest as bt
except Exception:
    bt = None  # plotting still works if you pass data directly


# -----------------------------------------------------------------------------
# Small analytics
# -----------------------------------------------------------------------------

def _rolling_sharpe(daily_ret: pd.Series, window: int = 63) -> pd.Series:
    dr = pd.Series(daily_ret).astype(float)
    mu = dr.rolling(window, min_periods=window).mean()
    sd = dr.rolling(window, min_periods=window).std(ddof=1)
    rs = mu / sd * np.sqrt(252)
    return rs.replace([np.inf, -np.inf], np.nan)


def _drawdown(cum_pnl: pd.Series) -> pd.Series:
    cp = pd.Series(cum_pnl).astype(float)
    peak = cp.cummax()
    dd = cp - peak
    return dd


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def make_all_plots(
    equity: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    *,
    outdir: str | os.PathLike = "../results/figures",
    title: str = "Stat‑Arb V6 Backtest",
    show: bool = False,
) -> Dict[str, str]:
    """Save equity, rolling Sharpe, and drawdown figures.

    Returns a dict of artifact paths.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Equity curve
    fig1 = plt.figure(figsize=(10, 4))
    ax1 = fig1.add_subplot(111)
    ax1.plot(equity.index, equity.values)
    ax1.set_title(f"{title} - Equity (Cum P&L)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cum P&L ($)")
    fig1.tight_layout()
    p1 = out / "equity_curve.png"
    fig1.savefig(p1, dpi=120)
    if show:
        plt.show()
    plt.close(fig1)

    # Rolling Sharpe (63d) based on daily returns
    daily_ret = equity.diff().fillna(equity)  # equity is cum P&L; diff → daily P&L
    daily_ret = daily_ret / max(abs(daily_ret).max(), 1.0)  # scale‑free if capital unknown
    rs = _rolling_sharpe(daily_ret, window=63)
    fig2 = plt.figure(figsize=(10, 3.5))
    ax2 = fig2.add_subplot(111)
    ax2.plot(rs.index, rs.values)
    ax2.set_title("Rolling Sharpe (63d)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sharpe")
    fig2.tight_layout()
    p2 = out / "rolling_sharpe_63d.png"
    fig2.savefig(p2, dpi=120)
    if show:
        plt.show()
    plt.close(fig2)

    # Drawdown
    dd = _drawdown(equity)
    fig3 = plt.figure(figsize=(10, 3.5))
    ax3 = fig3.add_subplot(111)
    ax3.fill_between(dd.index, dd.values, 0.0, step="pre")
    ax3.set_title("Drawdown (relative to running peak)")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Drawdown ($)")
    fig3.tight_layout()
    p3 = out / "drawdown.png"
    fig3.savefig(p3, dpi=120)
    if show:
        plt.show()
    plt.close(fig3)

    return {"equity": str(p1), "rolling_sharpe": str(p2), "drawdown": str(p3)}


# -----------------------------------------------------------------------------
# CLI entry (optional): run backtest then plot
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if bt is None:
        raise SystemExit("backtest module not available; run from project package or pass data to make_all_plots")
    res = bt.run()
    equity = res["equity"]
    trades = res.get("trades") if isinstance(res, dict) else None
    paths = make_all_plots(equity, trades, outdir="../results/figures", show=False)
    print("Saved:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
