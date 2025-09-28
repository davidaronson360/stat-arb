#!/usr/bin/env python3
"""
Generate 50 'zoomed' parameter sets for V6, adding ADF p-value threshold.

Outputs:
  results/tables/tune_cv_params_v6.csv
"""

from __future__ import annotations
import csv, random
from pathlib import Path
from typing import Dict, List, Sequence

# ---- Adjust if your code uses a different key for ADF threshold ----
ADF_KEY = "gates.adf_pvalue"   # change to the exact key your formation module reads
ADF_RANGE = (0.01, 0.20)          # typical p-value sweep; tweak if you like

OUTPUT = Path("results/tables/tune_cv_params_v6.csv")
N = 50
MASTER_SEED = 202_509_25  # reproducible

def weighted_choice(values: Sequence, favored: set) -> object:
    w = [2 if v in favored else 1 for v in values]
    return random.choices(list(values), weights=w, k=1)[0]

def make_row(i: int) -> Dict[str, object]:
    # V5 zoom space (from append_zoom_params.py)
    entry_z = random.uniform(1.7, 2.5)
    exit_z = random.uniform(0.05, 0.8)
    hard_stop_z = weighted_choice([5, 6], favored={5})
    fdr_alpha = random.uniform(0.06, 0.12)
    time_stop_multiple = weighted_choice([3.0, 4.0], favored={4.0})
    hl_min_days = 3
    hl_max_days = weighted_choice([60, 20, 30], favored={60})
    max_active_spreads = weighted_choice([3, 4, 5, 6, 7, 8], favored={6, 7, 8})

    # ADF threshold
    adf_alpha = random.uniform(*ADF_RANGE)

    run_id = f"{i:03d}"
    seed = 300_000 + i  # distinct from old runs

    row = {
        "run_id": run_id,
        "seed": seed,
        "signals.entry_z": round(entry_z, 4),
        "signals.exit_z": round(exit_z, 4),
        "signals.hard_stop_z": int(hard_stop_z),
        "gates.fdr_alpha": round(fdr_alpha, 4),
        "gates.time_stop_multiple": float(time_stop_multiple),
        "gates.hl_min_days": int(hl_min_days),
        "gates.hl_max_days": int(hl_max_days),
        "risk.max_active_spreads": int(max_active_spreads),
        ADF_KEY: round(adf_alpha, 4),
    }
    return row

def main():
    random.seed(MASTER_SEED)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [make_row(i) for i in range(N)]
    with OUTPUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {OUTPUT}")

if __name__ == "__main__":
    main()
