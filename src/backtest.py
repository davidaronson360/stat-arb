"""Backtesting pipeline for Stat-Arb V6: formation -> gating -> trading -> P&L aggregation."""

from __future__ import annotations

# --- IMPORTS (dual-mode: package or script) ---
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

if __package__ in (None, ""):
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    from universe import load_universe, Universe
    from data import load_prices
    from formation import discover_candidates, build_spread, SpreadSpec
    from gating import GateConfig, evaluate_candidates, fdr_bh
    from portfolio import PortfolioConfig, allocate, _legs_from_notional
    from costs import CostConfig, Trade, cost_per_trade
else:
    from .universe import load_universe, Universe
    from .data import load_prices
    from .formation import discover_candidates, build_spread, SpreadSpec
    from .gating import GateConfig, evaluate_candidates, fdr_bh
    from .portfolio import PortfolioConfig, allocate, _legs_from_notional
    from .costs import CostConfig, Trade, cost_per_trade
# --- END IMPORTS ---

logger = logging.getLogger("stat_arb_v6.backtest")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

logger.setLevel(logging.INFO)  # set to DEBUG to see HL/time-stop per-trade lines

# -----------------------------------------------------------------------------
# Config containers
# -----------------------------------------------------------------------------

@dataclass
class BTConfig:
    formation: int
    trade: int
    step: int
    entry_z: float
    exit_z: float
    hard_stop_z: float
    fdr_alpha: Optional[float] = None
    capital: float = 1_000_000.0
    # Execution/signals
    cost_edge_k: float = 1.8             # require expected edge >= k * round-trip costs
    act_at_next_bar: bool = False        # T+1 toggle (left False to avoid surprise)
    confirm_days: int = 1                # band confirmation (1 = none)
    time_stop_mult: float = 3.0          # time stop multiplier on half-life (e.g., 3 * HL)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _windows(index: pd.DatetimeIndex, formation: int, trade: int, step: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate rolling [formation, trade] windows over a price index.

    Returns a list of (formation_start, formation_end, trade_end) timestamps where:
    - `formation` is the lookback length (in bars)
    - `trade` is the forward evaluation window (in bars)
    - `step` advances the end of the formation window (in bars)
    """
    res: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for end_i in range(formation - 1, len(index) - 1, step):
        form_start = index[end_i - (formation - 1)]
        form_end = index[end_i]
        trade_end_i = min(end_i + trade, len(index) - 1)
        res.append((form_start, form_end, index[trade_end_i]))
    return res

def _load_cfg(config_path: str | Path) -> Tuple[Universe, BTConfig, GateConfig, PortfolioConfig, CostConfig]:
    """Parse YAML config and build typed configs for backtest components.
    
    This reads:
    - windows/signals/gates/execution/risk/costs blocks
    - universe (tickers and date bounds)
    
    Returns the Universe and config dataclasses used by the pipeline.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())

    u = load_universe(config_path)

    w = cfg.get("windows", {})
    signals = cfg.get("signals", {})
    gates = cfg.get("gates", {})
    entry_filter = cfg.get("entry_filter", {})
    execution = cfg.get("execution", {})
    rcfg = cfg.get("risk", {})
    ccfg = cfg.get("costs", {})

    bt = BTConfig(
        formation=int(w.get("formation", 252)),
        trade=int(w.get("trade", 63)),
        step=int(w.get("step", 21)),
        entry_z=float(signals.get("entry_z", 2.0)),
        exit_z=float(signals.get("exit_z", 0.5)),
        hard_stop_z=float(signals.get("hard_stop_z", 4.0)),
        fdr_alpha=float(gates.get("fdr_alpha")) if gates.get("fdr_alpha") is not None else None,
        capital=float(rcfg.get("capital", 1_000_000.0)),
        cost_edge_k=float(entry_filter.get("cost_edge_k", 1.8)),
        act_at_next_bar=bool(execution.get("act_at_next_bar", False)),
        confirm_days=int(signals.get("confirm_days", 1)),
        time_stop_mult=float(gates.get("time_stop_multiple", 3.0)),
    )

    gcfg = GateConfig(
        adf_alpha=float(gates.get("adf_pvalue", 0.05)),
        half_life_bounds=tuple(gates.get("half_life", [5, 60])),
        require_negative_adf_slope=True,
        require_rho_in_0_1=True,
        rho_signif_alpha=0.05,
        use_log_prices=True,
    )

    pcfg = PortfolioConfig(
        target_vol_annual=float(rcfg.get("target_vol_annual", 0.10)),
        max_gross_leverage=float(rcfg.get("max_gross_leverage", 2.0)),
        max_active_spreads=int(rcfg.get("max_active_spreads", 10)),
        sigma_window=63,
    )

    ccf = CostConfig(
        commission_per_share=float(ccfg.get("commission_per_share", 0.001)),
        half_spread_ticks=float(ccfg.get("half_spread_ticks", 1)),
        tick_size=0.01,
        slippage_bps=float(ccfg.get("slippage_bps", 2)),
    )

    return u, bt, gcfg, pcfg, ccf

def _find_exit_with_time_stop(
    z: pd.Series,
    trade_days: pd.DatetimeIndex,
    entry_idx: pd.Timestamp,
    exit_z: float,
    direction: int,
    hard_stop_z: float,
    time_stop_days: int,
    min_hold_days: int = 0,  # optional guardrail against 1-day churn
) -> Tuple[pd.Timestamp, str, int]:
    """
    Find the first exit after `entry_idx` according to these rules, in order:
      - Profit band:        |z| < exit_z               -> "band"
      - Directional stop:   adverse |z| > hard_stop_z  -> "hard_stop"
      - Time stop:          days_held ≥ time_stop_days -> "time_stop"
      - Otherwise:          end of window              -> "end_of_window"
    Returns (exit_idx, reason, days_held).
    """
    pos = trade_days.get_loc(entry_idx)
    if pos + 1 >= len(trade_days):
        return entry_idx, "end_of_window", 0

    days_held = 0
    for ts in trade_days[pos + 1:]:
        days_held += 1
        zt = float(z.get(ts, np.nan))
        if np.isfinite(zt):
            s = 1 if direction > 0 else -1      # +1 if entry on negative z (long), -1 if entry on positive z (short)
            signed = s * zt                      # negative at entry; moves toward 0 is favorable

            # Fast profit take if we cross through zero
            if signed >= 0.0:
                return ts, "band", days_held

            # Profit band on the favorable side (respects min_hold_days)
            if signed >= -exit_z and days_held >= min_hold_days:
                return ts, "band", days_held

            # Directional hard stop (adverse only)
            if signed <= -hard_stop_z and days_held >= min_hold_days:
                return ts, "hard_stop", days_held

        if days_held >= max(min_hold_days, int(np.ceil(time_stop_days))):
            return ts, "time_stop", days_held

    return trade_days[-1], "end_of_window", days_held

# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------

def run(config_path: str = "../configs/base.yaml") -> Dict[str, object]:
    config_path = str(config_path)  # allow Path inputs
    u, bt, gcfg, pcfg, ccf = _load_cfg(config_path)

    # Resolve z-source (simple read; you already control robust via formation stats below)
    try:
        y = yaml.safe_load(Path(config_path).read_text())
        z_source = str(y.get("signals", {}).get("z_source", "standard")).lower()
    except Exception:
        z_source = "standard"
    logger.info(f"[CONFIG] z_source={z_source} | cost_edge_k={bt.cost_edge_k:.2f} | act_at_next_bar={bt.act_at_next_bar} | confirm_days={bt.confirm_days} | time_stop_mult={bt.time_stop_mult:.2f}")

    px = load_prices(u.tickers(), u.start, u.end)
    if px.empty:
        raise RuntimeError("No price data loaded; check tickers and date range")

    wins = _windows(px.index, bt.formation, bt.trade, bt.step)

    # Trade logging helpers
    trade_seq = 0  # unique id per round-trip
    ind_map = {t: ind for ind, ts in u.by_industry().items() for t in ts}

    equity_gross = pd.Series(0.0, index=px.index)
    equity_net   = pd.Series(0.0, index=px.index)
    all_trades: List[Dict[str, object]] = []
    total_costs = 0.0
    total_cost_skipped = 0

    # Global exit reason table
    global_exit_reasons: Dict[str, int] = {}

    for w_i, (t0, tF, tT) in enumerate(wins, start=1):
        form = px.loc[t0:tF]
        trade = px.loc[tF:tT]
        if len(form) < bt.formation or len(trade) < 2:
            continue

        inds = u.by_industry()
        cands = discover_candidates(form, inds)
        if not cands:
            logger.info("Step %3d | cand=%3d  pass_ind=%3d  pass_fdr=%3d  selected=%2d  executed=%2d  cost_skipped=%2d | gross=%10.2f  net=%10.2f",
                        w_i, 0, 0, 0, 0, 0, 0, 0.0, 0.0)
            continue

        # ---------- Evaluate gates
        gated = evaluate_candidates(form, cands, gcfg)

        # Collect ADF pvals for FDR
        def _p(gr):
            try: return float(gr.diag.adf_pvalue)
            except Exception: pass
            try: return float(gr.adf_pvalue)
            except Exception: pass
            return np.nan

        p_all = np.array([_p(gr) for gr in gated], dtype=float)
        valid = np.isfinite(p_all)

        if not valid.any():
            logger.info("Step %3d | cand=%3d  pass_ind=%3d  pass_fdr=%3d  selected=%2d  executed=%2d  cost_skipped=%2d | gross=%10.2f  net=%10.2f",
                        w_i, len(cands), 0, 0, 0, 0, 0, 0.0, 0.0)
            continue

        if bt.fdr_alpha is not None:
            keep_valid, _ = fdr_bh(p_all[valid], alpha=float(bt.fdr_alpha))
            fdr_keep = np.zeros(len(gated), dtype=bool)
            fdr_keep[np.where(valid)[0]] = keep_valid
        else:
            fdr_keep = np.ones(len(gated), dtype=bool)

        passed = [(c, gr) for (c, gr, k) in zip(cands, gated, fdr_keep) if gr.passed and k]
        n_cand = len(cands)
        n_pass_ind = int(sum(gr.passed for gr in gated))
        n_pass_fdr = len(passed)
        if not passed:
            logger.info("Step %3d | cand=%3d  pass_ind=%3d  pass_fdr=%3d  selected=%2d  executed=%2d  cost_skipped=%2d | gross=%10.2f  net=%10.2f",
                        w_i, n_cand, n_pass_ind, 0, 0, 0, 0, 0.0, 0.0)
            continue

        # Select top by discovery score
        passed.sort(key=lambda cr: float(cr[0].score))
        sel = passed[: pcfg.max_active_spreads]
        n_selected = len(sel)

        # Map each selected spec to its GateResult (to access half-life later)
        gate_map: Dict[SpreadSpec, object] = {c.spec: gr for (c, gr) in sel}

        # ---------- Build spreads (log) and size notionals
        P_log = np.log(px.loc[t0:tT].clip(lower=1e-6))
        siz_map: Dict[SpreadSpec, pd.Series] = {}
        for cand, _ in sel:
            cols = list(cand.spec.tickers)
            s_full = build_spread(P_log[cols], cand.spec)
            siz_map[cand.spec] = s_full.loc[t0:tF]

        last_prices_start = form.iloc[-1]
        alloc = allocate(spread_series=siz_map, last_prices=last_prices_start, capital=bt.capital, cfg=pcfg)
        if not alloc.sizings:
            logger.info("Step %3d | cand=%3d  pass_ind=%3d  pass_fdr=%3d  selected=%2d  executed=%2d  cost_skipped=%2d | gross=%10.2f  net=%10.2f",
                        w_i, n_cand, n_pass_ind, n_pass_fdr, n_selected, 0, 0, 0.0, 0.0)
            continue

        trade_days = trade.index
        day_pnl_gross = pd.Series(0.0, index=trade_days)
        day_pnl_net   = pd.Series(0.0, index=trade_days)
        n_executed = 0
        n_cost_skipped = 0
        exit_reason_counts: Dict[str, int] = {}

        for sz in alloc.sizings:
            spec = sz.spec
            cols = list(spec.tickers)
            s_full = build_spread(P_log[cols], spec)
            s_form = s_full.loc[t0:tF]
            if s_form.notna().sum() < 30:
                continue

            # Formation stats: robust if requested
            if z_source == "robust":
                med = float(s_form.median())
                mad = float((s_form - med).abs().median())
                if np.isfinite(mad) and mad > 0.0:
                    mu = med
                    sd = 1.4826 * mad
                else:
                    mu = float(s_form.mean())
                    sd = float(s_form.std(ddof=1))
            else:
                mu = float(s_form.mean())
                sd = float(s_form.std(ddof=1))

            if not (np.isfinite(sd) and sd > 0.0):
                continue

            # Trade-window z from formation stats
            s_trade = s_full.loc[tF:tT]
            z = (s_trade - mu) / sd

            # Entry scan with confirmation
            entry_idx = None
            direction = 0
            streak = 0
            for ts, zt in z.items():
                if abs(zt) > bt.entry_z:
                    streak += 1
                    if streak >= bt.confirm_days:
                        direction = 1 if zt < 0 else -1
                        entry_idx = ts
                        break
                else:
                    streak = 0
            if entry_idx is None:
                continue

            # Optional T+1 execution (use next bar if available)
            if bt.act_at_next_bar:
                pos = trade_days.get_loc(entry_idx)
                if pos + 1 >= len(trade_days):
                    continue
                entry_idx = trade_days[pos + 1]

            # Cost-aware entry
            entry_prices = px.loc[entry_idx, cols]
            legs_est = _legs_from_notional(spec, notional=direction * sz.notional, last_prices=entry_prices)

            # Use formation window to estimate this spread's daily $ vol with current shares
            p_form_cap = px.loc[t0:tF, cols]
            shares_tmp = pd.Series({l.ticker: l.shares for l in legs_est})
            pnl_form_tmp = p_form_cap.diff().iloc[1:].mul(shares_tmp, axis=1).sum(axis=1)
            sd_pnl_dollars_tmp = float(pnl_form_tmp.std(ddof=1))
            
            # Allowable daily $ vol per spread from portfolio target
            sigma_port_daily = bt.capital * pcfg.target_vol_annual / np.sqrt(252.0)
            sigma_cap_each   = sigma_port_daily / max(1, pcfg.max_active_spreads)
            
            scale_k = 1.0
            if np.isfinite(sd_pnl_dollars_tmp) and sd_pnl_dollars_tmp > 0.0:
                scale_k = min(1.0, sigma_cap_each / sd_pnl_dollars_tmp)
            
            # If this trade is too "hot" shrink notional and rebuild legs
            if scale_k < 0.999:
                legs_est = _legs_from_notional(
                    spec,
                    notional=direction * sz.notional * scale_k,
                    last_prices=entry_prices,
                )

            # Approximate round-trip cost using costs at entry prices for both sides
            entry_trade_est = Trade(when=entry_idx, legs=tuple(legs_est))
            cb_entry_est = cost_per_trade(entry_trade_est, ccf)
            cb_exit_est  = cost_per_trade(entry_trade_est, ccf)  # symmetric approx
            round_trip_cost = cb_entry_est.total + cb_exit_est.total

            # Current z at entry
            z_now = float(z.loc[entry_idx])

            # --- Dollar-based expected edge to the band, using THIS trade's shares
            p_form = px.loc[t0:tF, cols]
            shares = pd.Series({l.ticker: l.shares for l in legs_est})
            leg_shares_str = ";".join(f"{t}:{int(shares[t])}" for t in cols)
            pnl_form = p_form.diff().iloc[1:].mul(shares, axis=1).sum(axis=1)  # $/day with these shares
            sd_pnl_dollars = float(pnl_form.std(ddof=1))
            sigmas_to_band = max(0.0, abs(z_now) - bt.exit_z)
            expected_gross = sigmas_to_band * sd_pnl_dollars if (np.isfinite(sd_pnl_dollars) and sd_pnl_dollars > 0) else 0.0

            if expected_gross < bt.cost_edge_k * round_trip_cost:
                n_cost_skipped += 1
                total_cost_skipped += 1
                continue  # do not enter

            # ---- Enter
            n_executed += 1
            entry_trade = Trade(when=entry_idx, legs=tuple(legs_est))
            cb_entry = cost_per_trade(entry_trade, ccf)
            total_costs += cb_entry.total

            trade_seq += 1
            tid = trade_seq
            spec_key = "|".join(cols)
            spec_inds = sorted({ind_map.get(t, "UNK") for t in cols})
            spec_industry = "|".join(spec_inds)

            all_trades.append({
                "trade_id": tid,
                "when": entry_idx,
                "action": "ENTER",
                "spec_key": spec_key,
                "tickers": list(cols),
                "industry": spec_industry,
                "dir": int(direction),
                "z_entry": float(z_now),
                "mu": float(mu),
                "sd": float(sd),
                "hl_days": float("nan"),  # filled below if available
                "time_stop_days": int(0), # filled below
                "notional": float(sz.notional),
                "entry_when": entry_idx,
                "entry_cost": cb_entry.total,
                "commission": cb_entry.commission,
                "half_spread": cb_entry.half_spread,
                "slippage": cb_entry.slippage,
                "total_cost": cb_entry.total,
                "leg_shares": leg_shares_str
            })
            # Apply entry costs on the entry bar
            if entry_idx in day_pnl_net.index:
                day_pnl_net.at[entry_idx] -= cb_entry.total

            # ---- Compute time-stop days from half-life (from gate diagnostics)
            gr = gate_map.get(spec, None)

            hl_days = np.nan
            if gr is not None and getattr(gr, "diag", None) is not None:
                for attr in ("half_life", "half_life_days", "hl_days"):
                    try:
                        v = float(getattr(gr.diag, attr))
                        if np.isfinite(v) and v > 0:
                            hl_days = v
                            break
                    except Exception:
                        pass

            if np.isfinite(hl_days) and hl_days > 0:
                time_stop_days = int(np.ceil(bt.time_stop_mult * hl_days))
            else:
                # Fallback: if gating computed an absolute time stop, use it.
                tstop_fallback = np.nan
                if gr is not None and getattr(gr, "diag", None) is not None:
                    try:
                        tstop_fallback = float(getattr(gr.diag, "time_stop_days"))
                    except Exception:
                        tstop_fallback = np.nan
                time_stop_days = int(np.ceil(tstop_fallback)) if (np.isfinite(tstop_fallback) and tstop_fallback > 0) else int(1e9)

            # Debug line for HL/time-stop choice
            if logger.isEnabledFor(logging.DEBUG):
                _used_hl = float(hl_days) if np.isfinite(hl_days) and hl_days > 0 else np.nan
                logger.debug(f"{spec.tickers} | HL={_used_hl:.2f} days | tstop_mult={bt.time_stop_mult:.2f} -> time_stop_days={int(time_stop_days)}")

            # ---- Decide the exit index & reason using z + time-stop
            exit_idx, exit_reason, days_held = _find_exit_with_time_stop(
                z=z,
                trade_days=trade_days,
                entry_idx=entry_idx,
                exit_z=bt.exit_z,
                hard_stop_z=bt.hard_stop_z,
                time_stop_days=time_stop_days,
                direction=direction, 
                min_hold_days=0,
            )
            exit_reason_counts[exit_reason] = exit_reason_counts.get(exit_reason, 0) + 1

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{spec.tickers} | EXIT reason={exit_reason} | days_held={int(days_held)} | entry={entry_idx.date()} -> exit={exit_idx.date()}")

            # ---- Accrue P&L day-by-day from entry to exit (price P&L only; costs handled on entry/exit bars)
            prev = entry_idx
            for ts in trade_days[trade_days.get_loc(entry_idx):]:
                if ts == entry_idx:
                    continue
                p_today = px.loc[ts, cols]
                p_prev  = px.loc[prev, cols]
                pnl = float(sum(l.shares * (p_today[l.ticker] - p_prev[l.ticker]) for l in legs_est))
                day_pnl_gross.loc[ts] += pnl
                day_pnl_net.loc[ts]   += pnl
                prev = ts
                if ts == exit_idx:
                    break

            # ---- Exit (apply costs on the chosen exit index; compute trade-level P&L)
            exit_trade = Trade(when=exit_idx, legs=tuple(legs_est))
            cb_exit = cost_per_trade(exit_trade, ccf)
            total_costs += cb_exit.total

            # realized gross/net for the round-trip
            p_entry = px.loc[entry_idx, cols]
            p_exit  = px.loc[exit_idx,  cols]
            gross_trade_pnl = float(sum(l.shares * (p_exit[l.ticker] - p_entry[l.ticker]) for l in legs_est))
            net_trade_pnl   = gross_trade_pnl - (cb_entry.total + cb_exit.total)
            
            # --- INVARIANTS: exit routing must be sane ---
            z_exit_val = float(z.get(exit_idx, np.nan))
            s_z_exit_val = float(direction * z_exit_val)
            _eps = 1e-9  # numerical wiggle room
            
            # 1) If we say "hard_stop", the directional threshold must be hit (adverse only)
            if exit_reason == "hard_stop":
                assert s_z_exit_val <= -bt.hard_stop_z + _eps, (
                    f"Hard stop without hitting directional threshold: "
                    f"s*z_exit={s_z_exit_val:.3f}, threshold={-bt.hard_stop_z:.3f}"
                )            
                # Net P&L can be >0 here (path-dependent). Log it for awareness instead of failing.
                if net_trade_pnl > 0.0:
                    logger.warning(
                        f"Hard stop on net winner | trade_id={tid} | s*z_exit={s_z_exit_val:.3f} "
                        f"| net_pnl={net_trade_pnl:.2f} | reason={exit_reason}"
                    )
            
            # 2) If we crossed through zero (favorable side), we must have exited via band
            if s_z_exit_val >= 0.0 - _eps:
                assert exit_reason == "band", (
                    f"Crossed zero but exit_reason='{exit_reason}' (expected 'band'); "
                    f"s*z_exit={s_z_exit_val:.3f}"
                )
            
            all_trades.append({
                "trade_id": tid,
                "when": exit_idx,
                "action": "EXIT",
                "spec_key": spec_key,
                "tickers": list(cols),
                "industry": spec_industry,
            
                # Fields duplicated here so trades_roundtrip.csv has no NaNs
                "dir": int(direction),
                "z_entry": float(z_now),
                "leg_shares": leg_shares_str,
                "entry_cost": cb_entry.total,
            
                # --- Exit info
                "exit_reason": exit_reason,
                "days_held": int(days_held),
                "z_exit": float(z.get(exit_idx, np.nan)),
            
                # --- Diagnostics for audits (signed by entry direction)
                "s_z_entry": float(direction * z_now),                      # should be < 0 at entry
                "s_z_exit": float(direction * float(z.get(exit_idx, np.nan))),
                "crossed_zero": bool(direction * float(z.get(exit_idx, np.nan)) >= 0.0),
                "stop_dir_hit": bool(
                    (exit_reason == "hard_stop") and
                    (direction * float(z.get(exit_idx, np.nan)) <= -bt.hard_stop_z)
                ),
            
                # --- Stats carried through
                "mu": float(mu),
                "sd": float(sd),
                "hl_days": float(hl_days) if np.isfinite(hl_days) else np.nan,
                "time_stop_days": int(time_stop_days),
                "notional": float(sz.notional),
                "entry_when": entry_idx,
            
                # --- P&L and costs
                "gross_trade_pnl": gross_trade_pnl,
                "net_trade_pnl": net_trade_pnl,
                "exit_cost": cb_exit.total,
                "round_trip_cost": cb_entry.total + cb_exit.total,
                "commission": cb_exit.commission,
                "half_spread": cb_exit.half_spread,
                "slippage": cb_exit.slippage,
                "total_cost": cb_exit.total,
            })

            # Cost hits exactly on the exit bar
            if exit_idx in day_pnl_net.index:
                day_pnl_net.at[exit_idx] -= cb_exit.total

        # ---- Accumulate window P&L
        equity_gross.loc[trade_days] += day_pnl_gross
        equity_net.loc[trade_days]   += day_pnl_net

        # roll up step exit-reason counts into global
        for k, v in exit_reason_counts.items():
            global_exit_reasons[k] = global_exit_reasons.get(k, 0) + v

        er_str = " ".join(f"{k}:{v}" for k, v in exit_reason_counts.items()) if exit_reason_counts else "-"
        logger.info(
            "Step %3d | cand=%3d  pass_ind=%3d  pass_fdr=%3d  selected=%2d  executed=%2d  cost_skipped=%2d | gross=%10.2f  net=%10.2f | exits={%s}",
            w_i, n_cand, n_pass_ind, n_pass_fdr, n_selected, n_executed, n_cost_skipped,
            float(day_pnl_gross.sum()), float(day_pnl_net.sum()), er_str
        )

    # ---- Summary
    ret_gross = equity_gross / bt.capital
    ret_net   = equity_net   / bt.capital

    equity_cum_gross = equity_gross.cumsum()
    equity_cum_net   = equity_net.cumsum()

    exit_table = dict(sorted(global_exit_reasons.items(), key=lambda kv: (-kv[1], kv[0])))

    trades_df = pd.DataFrame(all_trades)
    total_trades = int((trades_df["action"] == "ENTER").sum()) if not trades_df.empty else 0
    skip_rate = total_cost_skipped / max(1, total_cost_skipped + total_trades)

    summary = {
        "Gross APR": float(_apr(ret_gross)),
        "Gross Sharpe": float(_sharpe(ret_gross)),
        "Net APR": float(_apr(ret_net)),
        "Net Sharpe": float(_sharpe(ret_net)),
        "Net Sharpe (excess Rf)": float(sharpe_excess(ret_net, rf=0.04, rf_mode="apr")),
        "Final Gross P&L": round(float(equity_cum_gross.iloc[-1]), 2),
        "Final Net P&L": round(float(equity_cum_net.iloc[-1]), 2),
        "Total trades": total_trades,
        "Total costs (applied)": float(total_costs),
        "Cost-skipped entries": int(total_cost_skipped),
        "Cost skip rate": float(skip_rate),
        "Exit reasons": exit_table,
    }

    logger.info(
        "Portfolio | Steps=%d | Trades=%d | Gross APR: %.2f%%  Net APR: %.2f%% | "
        "Gross Sharpe: %.2f  Net Sharpe: %.2f  Excess Net Sharpe: %.2f | "
        "Gross P&L: %.2f  Net P&L: %.2f",
        len(wins),
        summary["Total trades"],
        100 * summary["Gross APR"],
        100 * summary["Net APR"],
        summary["Gross Sharpe"],
        summary["Net Sharpe"],
        summary["Net Sharpe (excess Rf)"],   
        summary["Final Gross P&L"],
        summary["Final Net P&L"],
    )

    if exit_table:
        logger.info("Exit reasons summary:")
        total_exits = sum(exit_table.values())
        for k, v in exit_table.items():
            pct = 100.0 * v / max(1, total_exits)
            logger.info("  - %-12s : %5d  (%5.1f%%)", k, v, pct)

    # ---------- Save logs for analysis ----------
    outdir = Path("../results/tables")
    outdir.mkdir(parents=True, exist_ok=True)

    if not trades_df.empty:
        trades_df = trades_df.sort_values(["when", "action"])
        events_path_csv = outdir / "trade_events.csv"
        trades_df.to_csv(events_path_csv, index=False)

        # One row per completed trade = EXIT rows have all round-trip metrics
        roundtrip_cols = [
            "trade_id","entry_when","when","tickers","industry","dir",
            "leg_shares",
            "z_entry","z_exit","s_z_entry","s_z_exit","crossed_zero","stop_dir_hit",
            "mu","sd","hl_days","time_stop_days",
            "days_held","exit_reason","notional",
            "entry_cost","exit_cost","round_trip_cost","gross_trade_pnl","net_trade_pnl"
        ]
        roundtrips = trades_df.loc[trades_df["action"]=="EXIT", roundtrip_cols].rename(columns={"when":"exit_when"})
        roundtrips_path_csv = outdir / "trades_roundtrip.csv"
        roundtrips.to_csv(roundtrips_path_csv, index=False)

        try:
            trades_df.to_parquet(outdir / "trade_events.parquet", index=False)
            roundtrips.to_parquet(outdir / "trades_roundtrip.parquet", index=False)
        except Exception:
            pass  # Parquet optional

    return {
        "equity_gross": equity_cum_gross,
        "equity_net": equity_cum_net,
        "equity": equity_cum_net,
        "trades": trades_df,
        "summary": summary,
    }

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def _apr(daily_ret: pd.Series, *, trading_days: int = 252) -> float:
    """Annualized return from daily simple returns."""
    m = float(daily_ret.mean())
    return m * trading_days

def _sharpe(daily_ret: pd.Series, *, trading_days: int = 252) -> float:
    """Plain Sharpe vs zero benchmark (no risk-free)."""
    mu = float(daily_ret.mean())
    sd = float(daily_ret.std(ddof=1))
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return (mu / sd) * np.sqrt(trading_days)

# --- Conservative (excess) Sharpe over risk-free -----------------------------
def sharpe_excess(
    daily_net_ret: pd.Series,
    rf: float | pd.Series | None = None,
    *,
    trading_days: int = 252,
    rf_mode: str = "apr",  # "apr" for annualized rate; "daily" for daily simple rate
) -> float:
    """Sharpe of excess net returns (after costs, minus risk-free)."""
    if rf is None:
        # Fall back to raw Sharpe on net returns (what you have today)
        mu = float(daily_net_ret.mean())
        sd = float(daily_net_ret.std(ddof=1))
        return (mu / sd) * np.sqrt(trading_days) if np.isfinite(sd) and sd > 0 else 0.0

    if isinstance(rf, (int, float)):
        if rf_mode == "apr":
            daily_rf = (1.0 + float(rf)) ** (1.0 / trading_days) - 1.0
        else:
            daily_rf = float(rf)
        excess = daily_net_ret - daily_rf
    else:
        # Series: align and convert if needed
        rf_series = rf.reindex(daily_net_ret.index).ffill()
        if rf_mode == "apr":
            daily_rf_series = (1.0 + rf_series) ** (1.0 / trading_days) - 1.0
        else:
            daily_rf_series = rf_series
        excess = daily_net_ret - daily_rf_series

    mu = float(excess.mean())
    sd = float(excess.std(ddof=1))
    return (mu / sd) * np.sqrt(trading_days) if np.isfinite(sd) and sd > 0 else 0.0


__all__ = ["run"]

if __name__ == "__main__":
    res = run()
    try:
        if __package__ in (None, ""):
            import sys, pathlib
            sys.path.append(str(pathlib.Path(__file__).resolve().parent))
            import plots as pl
        else:
            from . import plots as pl
        pl.make_all_plots(res["equity_net"], res.get("trades"), outdir="../results/figures/net", title="Stat-Arb V6 Backtest (Net)", show=True)
        pl.make_all_plots(res["equity_gross"], res.get("trades"), outdir="../results/figures/gross", title="Stat-Arb V6 Backtest (Gross)", show=False)
    except Exception as e:
        print(f"Plotting skipped: {e}")
