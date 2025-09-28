#!/usr/bin/env python3
"""
Generate 100 random (but sensible) parameter sets for your stat-arb backtest.

- Plain RNG (no Sobol/LHS).
- Chooses ranges/choices so constraints are naturally satisfied:
  * signals.exit_z ∈ [0.0, 1.0] and signals.entry_z ∈ [1.2, 3.0] ⇒ exit < entry
  * gates.hl_min_days from small set; gates.hl_max_days from larger set ⇒ hl_max > hl_min
- Outputs CSV: results/tables/tune_global_params.csv
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List

# -----------------------------
# Config
# -----------------------------
N_RUNS = 100
OUTPUT_DIR = Path("results/tables")
OUTPUT_FILE = OUTPUT_DIR / "tune_global_params.csv"

# Fixed master seed for reproducible generation of the parameter file
MASTER_SEED = 42

# Reasonable ranges/choices (tweak if you like)
ENTRY_Z_RANGE = (1.2, 3.0)                 # signals.entry_z
EXIT_Z_RANGE = (0.0, 1.0)                  # signals.exit_z
HARD_STOP_CHOICES = [4, 5, 6]              # signals.hard_stop_z
FDR_ALPHA_RANGE = (0.05, 0.20)             # gates.fdr_alpha
TIME_STOP_CHOICES = [2.0, 2.5, 3.0, 3.5, 4.0]  # gates.time_stop_multiple
HL_MIN_CHOICES = [3, 5, 7, 10]             # gates.hl_min_days
HL_MAX_CHOICES = [20, 30, 45, 60]          # gates.hl_max_days
MAX_ACTIVE_SPREADS_CHOICES = [3, 4, 5, 6, 8, 10]  # risk.max_active_spreads


def generate_one(run_id: int) -> Dict[str, object]:
    """
    Generate one parameter row using plain RNG.
    We also give each row its own deterministic 'seed' based on run_id.
    """
    entry_z = random.uniform(*ENTRY_Z_RANGE)
    exit_z = random.uniform(*EXIT_Z_RANGE)  # always < entry_z because ranges don’t overlap
    hard_stop_z = random.choice(HARD_STOP_CHOICES)
    fdr_alpha = random.uniform(*FDR_ALPHA_RANGE)
    time_stop_multiple = random.choice(TIME_STOP_CHOICES)
    hl_min = random.choice(HL_MIN_CHOICES)
    hl_max = random.choice(HL_MAX_CHOICES)  # from a set that’s necessarily > hl_min
    max_active_spreads = random.choice(MAX_ACTIVE_SPREADS_CHOICES)

    # Per-run deterministic seed (useful if your backtest uses randomness)
    per_run_seed = 10_000 + run_id

    return {
        "run_id": f"{run_id:03d}",
        "seed": per_run_seed,
        # YAML-aligned keys for your overlay:
        "signals.entry_z": round(entry_z, 4),
        "signals.exit_z": round(exit_z, 4),
        "signals.hard_stop_z": int(hard_stop_z),
        "gates.fdr_alpha": round(fdr_alpha, 4),
        "gates.time_stop_multiple": float(time_stop_multiple),
        "gates.hl_min_days": int(hl_min),
        "gates.hl_max_days": int(hl_max),
        "risk.max_active_spreads": int(max_active_spreads),
    }


def main() -> None:
    random.seed(MASTER_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for i in range(N_RUNS):
        rows.append(generate_one(i))

    fieldnames = list(rows[0].keys())
    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} parameter sets to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
