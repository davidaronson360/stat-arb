#!/usr/bin/env python3
"""
Param sweep runner (Step 2).

- Reads:  results/tables/tune_global_params.csv   (from Step 1)
- Base:   configs/base.yaml (override with --base if needed)
- Calls:  backtest.run(config_path) via importlib
- Writes: results/tables/tune_global_results.csv  (appends; resumable)

Usage examples:
  # Run first 1 unfinished row
  python sweep_runner.py --limit 1

  # Run first 10 unfinished rows
  python sweep_runner.py --limit 10

  # Run specific run_ids
  python sweep_runner.py --run-ids 000,001,002

  # Run a range of run_ids
  python sweep_runner.py --run-ids 010-019

  # Custom paths
  python sweep_runner.py --params results/tables/tune_global_params.csv --base configs/base.yaml --backtest backtest.py
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

# ---- Defaults (override via CLI) ----
DEFAULT_PARAMS_CSV = Path("results/tables/tune_global_params.csv")
DEFAULT_RESULTS_CSV = Path("results/tables/tune_global_results.csv")
DEFAULT_BASE_YAML = Path("configs/base.yaml")      # change if your base is elsewhere
DEFAULT_BACKTEST_PY = Path("src/backtest.py")


def _read_params(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _read_done_ids(results_csv: Path) -> set:
    if not results_csv.exists():
        return set()
    with results_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return {row["run_id"] for row in reader if "run_id" in row}


def _parse_run_ids(arg: str) -> List[str]:
    """
    Accepts forms like: "000,001,007" or "010-019" or mixed "000,010-014,099".
    Returns zero-padded 3-char strings.
    """
    out = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a, b = int(a), int(b)
            out.extend([f"{k:03d}" for k in range(min(a, b), max(a, b) + 1)])
        else:
            out.append(f"{int(chunk):03d}")
    return out


def _nested_set(cfg: Dict, dotted_key: str, value) -> None:
    """Set cfg['a']['b'] = value for dotted_key='a.b' (creates subdicts if needed)."""
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _make_overlay_cfg(base_cfg: Dict, row: Dict[str, str]) -> Dict:
    """
    Build a new config dict by overlaying the base YAML with one row of params.
    Expects Step-1 columns:
      signals.entry_z, signals.exit_z, signals.hard_stop_z,
      gates.fdr_alpha, gates.time_stop_multiple, gates.hl_min_days, gates.hl_max_days,
      risk.max_active_spreads
    """
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy via round-trip

    # 1) direct dotted keys
    direct = [
        "signals.entry_z",
        "signals.exit_z",
        "signals.hard_stop_z",
        "gates.fdr_alpha",
        "gates.time_stop_multiple",
        "risk.max_active_spreads",
        "universe.start_date",
        "universe.end_date",
    ]
    for k in direct:
        if k in row and row[k] not in (None, "", "None"):
            v_raw = row[k]
            # cast numbers if possible
            try:
                v = float(v_raw)
                if v.is_integer():
                    v = int(v)
            except Exception:
                v = v_raw
            _nested_set(cfg, k, v)

    # 2) half-life pair from hl_min/hl_max
    hl_min = int(float(row.get("gates.hl_min_days")))
    hl_max = int(float(row.get("gates.hl_max_days")))
    cfg.setdefault("gates", {})
    cfg["gates"]["half_life"] = [hl_min, hl_max]

    return cfg

def _import_backtest(backtest_path: Path):
    from importlib.util import module_from_spec, spec_from_file_location
    import sys

    spec = spec_from_file_location("stat_arb_backtest", backtest_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import backtest at {backtest_path}")

    mod = module_from_spec(spec)
    # IMPORTANT: register in sys.modules so decorators (e.g., dataclasses) can resolve module globals
    sys.modules[spec.name] = mod

    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise RuntimeError("backtest.py does not expose a run(config_path: str) function")
    return mod

def _flatten_exit_reasons(er: Dict[str, int]) -> Dict[str, int]:
    # Standardize keys we care about; include unknowns with prefix
    out = {"exit_band": 0, "exit_time_stop": 0, "exit_hard_stop": 0, "exit_end_of_window": 0, "exit_other": 0}
    for k, v in (er or {}).items():
        kl = str(k).lower()
        if "band" in kl:
            out["exit_band"] += int(v)
        elif "time" in kl:
            out["exit_time_stop"] += int(v)
        elif "hard" in kl:
            out["exit_hard_stop"] += int(v)
        elif "end" in kl or "window" in kl:
            out["exit_end_of_window"] += int(v)
        else:
            out["exit_other"] += int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, default=DEFAULT_PARAMS_CSV, help="CSV from Step 1")
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS_CSV, help="Output results CSV")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE_YAML, help="Path to base.yaml")
    ap.add_argument("--backtest", type=Path, default=DEFAULT_BACKTEST_PY, help="Path to backtest.py")
    ap.add_argument("--limit", type=int, default=None, help="Max number of runs this invocation")
    ap.add_argument("--run-ids", type=str, default=None, help="Comma/range list like '000,001,010-019'")
    ap.add_argument("--universe-start", type=str, default=None, help="Override universe.start_date for this invocation (e.g., 2021-01-01)")
    ap.add_argument("--universe-end", type=str, default=None, help="Override universe.end_date for this invocation (e.g., 2022-12-31)")
    ap.add_argument("--tag-fold", type=str, default=None, help="Optional label to write into results (e.g., 2022)")
    args = ap.parse_args()

    params_rows = _read_params(args.params)
    done_ids = _read_done_ids(args.results)
    bt = _import_backtest(args.backtest)

    base_cfg = yaml.safe_load(args.base.read_text())

    # Select which run_ids to execute
    all_ids_in_csv = [r["run_id"] for r in params_rows]
    if args.run_ids:
        target_ids = set(_parse_run_ids(args.run_ids))
    else:
        target_ids = set(all_ids_in_csv)

    # Filter for remaining
    todo = [r for r in params_rows if r["run_id"] in target_ids and r["run_id"] not in done_ids]
    if args.limit is not None:
        todo = todo[: args.limit]

    # Ensure output dir exists
    args.results.parent.mkdir(parents=True, exist_ok=True)
    overlays_dir = Path("results/tmp/overlays")
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Prepare results writer (append or create)
    write_header = not args.results.exists()
    fieldnames = [
        "run_id", "seed",
        # params (echoed for convenience)
        "signals.entry_z", "signals.exit_z", "signals.hard_stop_z",
        "gates.fdr_alpha", "gates.time_stop_multiple",
        "gates.hl_min_days", "gates.hl_max_days",
        "risk.max_active_spreads",
        # metrics
        "Net Sharpe", "Net APR", "Gross Sharpe", "Gross APR",
        "Final Net P&L", "Final Gross P&L",
        "Total trades", "Total costs (applied)", "Cost-skipped entries", "Cost skip rate",
        # exits
        "exit_band", "exit_time_stop", "exit_hard_stop", "exit_end_of_window", "exit_other",
        # timing
        "seconds",
        # overlay path for reproducibility
        "overlay_yaml",
        "fold",
    ]

    with args.results.open("a", newline="") as rf:
        writer = csv.DictWriter(rf, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for row in todo:
            run_id = row["run_id"]
            t0 = time.time()
            try:
                # Build overlay YAML
                overlay_cfg = _make_overlay_cfg(base_cfg, row)
                if args.universe_start:
                    _nested_set(overlay_cfg, "universe.start_date", args.universe_start)
                if args.universe_end:
                    _nested_set(overlay_cfg, "universe.end_date", args.universe_end)

                overlay_path = (overlays_dir / f"overlay_{run_id}.yaml").resolve()
                with overlay_path.open("w") as yf:
                    yaml.safe_dump(overlay_cfg, yf, sort_keys=False)

                # Run the backtest (returns dict with 'summary')
                res = bt.run(str(overlay_path))
                summary = res.get("summary", {}) or {}

                exits = _flatten_exit_reasons(summary.get("Exit reasons") or {})
                seconds = round(time.time() - t0, 2)

                out_row = {
                    # identity
                    "run_id": run_id,
                    "seed": row.get("seed", ""),
                    # echo params for easy filtering later
                    "signals.entry_z": row.get("signals.entry_z", ""),
                    "signals.exit_z": row.get("signals.exit_z", ""),
                    "signals.hard_stop_z": row.get("signals.hard_stop_z", ""),
                    "gates.fdr_alpha": row.get("gates.fdr_alpha", ""),
                    "gates.time_stop_multiple": row.get("gates.time_stop_multiple", ""),
                    "gates.hl_min_days": row.get("gates.hl_min_days", ""),
                    "gates.hl_max_days": row.get("gates.hl_max_days", ""),
                    "risk.max_active_spreads": row.get("risk.max_active_spreads", ""),
                    # metrics (robust to missing keys)
                    "Net Sharpe": summary.get("Net Sharpe", ""),
                    "Net APR": summary.get("Net APR", ""),
                    "Gross Sharpe": summary.get("Gross Sharpe", ""),
                    "Gross APR": summary.get("Gross APR", ""),
                    "Final Net P&L": summary.get("Final Net P&L", ""),
                    "Final Gross P&L": summary.get("Final Gross P&L", ""),
                    "Total trades": summary.get("Total trades", ""),
                    "Total costs (applied)": summary.get("Total costs (applied)", ""),
                    "Cost-skipped entries": summary.get("Cost-skipped entries", ""),
                    "Cost skip rate": summary.get("Cost skip rate", ""),
                    # exits
                    **exits,
                    # timing + overlay
                    "seconds": seconds,
                    "overlay_yaml": str(overlay_path),
                    "fold": args.tag_fold or "",
                }

                writer.writerow(out_row)
                rf.flush()
                print(f"[OK] run_id={run_id}  Net Sharpe={out_row['Net Sharpe']}  seconds={seconds}")
            except Exception as e:
                print(f"[ERR] run_id={run_id}: {e}")
