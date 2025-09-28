#!/usr/bin/env python3
"""
V6 5-fold cross-validation runner (no CLI args needed).
- Reads:  results/tables/tune_cv_params_v6.csv
- Calls:  src/backtest.py: run(overlay_yaml_path) -> Dict
- Writes: results/tables/tune_cv_results_v6.csv
"""

from __future__ import annotations
import csv
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml

# ---------- DEFAULTS (edit here if your layout differs) ----------
PARAMS_CSV   = Path("results/tables/tune_cv_params_v6.csv")
RESULTS_CSV  = Path("results/tables/tune_cv_results_v6.csv")
BASE_YAML    = Path("configs/base.yaml")

# Load base.yaml just to capture industries
with BASE_YAML.open("r") as f:
    base_cfg = yaml.safe_load(f)
BASE_INDUSTRIES = base_cfg.get("universe", {}).get("industries", {})

BACKTEST_PY  = Path("src/backtest.py")
TMP_DIR      = Path("tmp/overlays")
# Optionally cap how many param rows to run while testing; set to None to run all 50
LIMIT_ROWS: int | None = None

# 5 folds (start of year → end of next year)
FOLDS: List[Tuple[str, str, str]] = [
    ("F1_2017_2018", "2017-01-01", "2018-12-31"),
    ("F2_2018_2019", "2018-01-01", "2019-12-31"),
    ("F3_2019_2020", "2019-01-01", "2020-12-31"),
    ("F4_2020_2021", "2020-01-01", "2021-12-31"),
    ("F5_2022_2023", "2022-01-01", "2023-12-31"),
]
# ---------------------------------------------------------------

TMP_DIR.mkdir(parents=True, exist_ok=True)

def load_backtest(backtest_path: Path):
    """Dynamic import: expects a function run(config_path: str|Path) -> Dict."""
    import sys, importlib.util

    spec = importlib.util.spec_from_file_location("backtest_v6", str(backtest_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import backtest from {backtest_path}")

    mod = importlib.util.module_from_spec(spec)  # type: ignore
    # IMPORTANT: register so decorators (dataclasses, etc.) can resolve module globals
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore

    if not hasattr(mod, "run"):
        raise RuntimeError("backtest.py must expose run()")
    return mod.run  # type: ignore

def read_param_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if LIMIT_ROWS is not None:
        rows = rows[:LIMIT_ROWS]
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    return rows

def write_overlay(overrides: Dict[str, Any], tag: str) -> Path:
    start = overrides.pop("__start_date")
    end   = overrides.pop("__end_date")

    overlay: Dict[str, Any] = {
        "universe": {
            "start_date": start,
            "end_date": end,
            "industries": BASE_INDUSTRIES,  # <- inject from base.yaml
        }
    }

    for k, v in overrides.items():
        cur = overlay
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v

    out = TMP_DIR / f"overlay_{tag}.yaml"
    with out.open("w") as f:
        yaml.safe_dump(overlay, f, sort_keys=False)
    return out

def ensure_headers(path: Path, headers: List[str]):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=headers).writeheader()

def pick_metrics(run_ret: Dict[str, Any]) -> Dict[str, Any]:
    """Collect a few standard metrics from either ret['summary'] or top-level."""
    out: Dict[str, Any] = {}
    for key in ("summary", "metrics"):
        if isinstance(run_ret.get(key), dict):
            out.update(run_ret[key])  # type: ignore
            break
    for k in ("Net Sharpe", "Gross Sharpe", "Net APR", "Gross APR", "Cost skip rate"):
        if k in run_ret and k not in out:
            out[k] = run_ret[k]
    return out

def smart_cast(x: str):
    for typ in (int, float):
        try:
            return typ(x)
        except Exception:
            pass
    return x

def main():
    run_fn = load_backtest(BACKTEST_PY)
    rows = read_param_rows(PARAMS_CSV)

    # Results header = base info + original param columns + metric columns
    param_cols = list(rows[0].keys())          # includes 'gates.adf_value'
    base_cols  = ["run_id", "seed", "fold", "start_date", "end_date", "seconds"]
    metric_cols = ["Net Sharpe", "Gross Sharpe", "Net APR", "Gross APR", "Cost skip rate"]
    headers = base_cols + param_cols + metric_cols
    ensure_headers(RESULTS_CSV, headers)

    with RESULTS_CSV.open("a", newline="") as rf:
        writer = csv.DictWriter(rf, fieldnames=headers, extrasaction="ignore")

        for row in rows:
            for fold_tag, start, end in FOLDS:
                tag = f"{row['run_id']}_{fold_tag}"
                # Prepare overrides (convert CSV strings to numeric where reasonable)
                overrides = {k: smart_cast(v) for k, v in row.items() if k not in ("run_id", "seed")}
                overrides["__start_date"] = start
                overrides["__end_date"]   = end

                overlay_path = write_overlay(overrides, tag)

                t0 = time.time()
                try:
                    ret = run_fn(overlay_path)  # your backtest merges base.yaml + overlay
                except Exception as e:
                    print(f"[ERR] run_id={row['run_id']} fold={fold_tag}: {e}")
                    continue
                secs = round(time.time() - t0, 3)

                metrics = pick_metrics(ret if isinstance(ret, dict) else {})
                out = {
                    "run_id": row["run_id"],
                    "seed": row["seed"],
                    "fold": fold_tag,
                    "start_date": start,
                    "end_date": end,
                    "seconds": secs,
                    **row,
                    **{k: metrics.get(k, "") for k in metric_cols},
                }
                writer.writerow(out)
                rf.flush()
                print(f"[OK] run_id={row['run_id']} {fold_tag} Net Sharpe={out['Net Sharpe']} secs={secs}")

if __name__ == "__main__":
    main()
