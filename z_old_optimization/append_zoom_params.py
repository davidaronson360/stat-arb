#!/usr/bin/env python3
"""
Append 50 zoomed parameter sets to results/tables/tune_global_params.csv.

Distributions (your spec):
- signals.entry_z ~ Uniform[1.7, 2.5]
- signals.exit_z  ~ Uniform[0.05, 0.8]
- signals.hard_stop_z ∈ {5, 6}          (5 weighted 2×)
- gates.fdr_alpha ~ Uniform[0.06, 0.12]
- gates.time_stop_multiple ∈ {3.0, 4.0} (4.0 weighted 2×)
- gates.hl_min_days = 3 (fixed)
- gates.hl_max_days ∈ {60, 20, 30}      (60 weighted 2×)
- risk.max_active_spreads ∈ {3,4,5,6,7,8} (6/7/8 weighted 2×)

Run IDs start at 200; seeds are integers that begin with "200" (e.g., 200200, 200201, ...).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence

# -------- settings you can tweak --------
PARAMS_CSV = Path("results/tables/tune_global_params.csv")
N_NEW = 50
START_RUN_ID = 200     # will use 200..(200+N_NEW-1), skipping any that already exist
MASTER_SEED = 202_409_16
# ----------------------------------------


def weighted_choice(values: Sequence, favored: set) -> object:
    """Pick with 2× weight for favored items, 1× otherwise."""
    weights = [2 if v in favored else 1 for v in values]
    return random.choices(list(values), weights=weights, k=1)[0]


def read_existing_ids(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Params CSV not found: {csv_path}")
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [row["run_id"] for row in reader if "run_id" in row]


def read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or []


def make_row(run_id_int: int) -> Dict[str, object]:
    # Uniforms
    entry_z = random.uniform(1.7, 2.5)
    exit_z = random.uniform(0.05, 0.8)
    fdr_alpha = random.uniform(0.06, 0.12)

    # Weighted choices
    hard_stop_z = weighted_choice([5, 6], favored={5})
    time_stop_multiple = weighted_choice([3.0, 4.0], favored={4.0})
    hl_min_days = 3
    hl_max_days = weighted_choice([60, 20, 30], favored={60})
    max_active_spreads = weighted_choice([3, 4, 5, 6, 7, 8], favored={6, 7, 8})

    run_id = f"{run_id_int:03d}"
    # seed that clearly starts with "200"
    seed = 200_000 + run_id_int

    return {
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
    }


def main():
    random.seed(MASTER_SEED)

    # Load header + existing IDs
    header = read_header(PARAMS_CSV)
    existing_ids = set(read_existing_ids(PARAMS_CSV))

    # Generate run_id sequence, skipping any collisions
    to_append: List[Dict[str, object]] = []
    rid = START_RUN_ID
    while len(to_append) < N_NEW:
        rid_str = f"{rid:03d}"
        if rid_str not in existing_ids:
            to_append.append(make_row(rid))
        rid += 1

    # Append in CSV order, ignoring any extra keys
    with PARAMS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header or list(to_append[0].keys()), extrasaction="ignore")
        for row in to_append:
            writer.writerow(row)

    print(f"Appended {len(to_append)} zoom rows to {PARAMS_CSV}")
    print(f"First new run_id: {to_append[0]['run_id']}, last: {to_append[-1]['run_id']}")


if __name__ == "__main__":
    main()
