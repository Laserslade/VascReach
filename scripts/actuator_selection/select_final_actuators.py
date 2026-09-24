import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

import numpy as np

import canonical as can
import actuator as act
from test_canonical import outlet_order_for

PRODUCTION_BUDGET = 100
PRODUCTION_SEED = 424242
FAMILIES = ["symmetric_tree", "murray_tree", "looped_hierarchical", "grid_lattice"]


def run():
    families, _ = can.build_all_families()
    screening = act.frozen_screening_demand_set()

    rows = []
    for fam in FAMILIES:
        G = families[fam]
        outlet_order = outlet_order_for(G)
        t0 = time.time()
        result = act.select_placement_nonlinear(
            G, outlet_order, screening, budget=PRODUCTION_BUDGET, seed=PRODUCTION_SEED
        )
        elapsed = time.time() - t0
        rows.append({
            "family": fam,
            "candidate": str(result["candidate"]),
            "mean_error": result["mean_error"],
            "n_evaluated": result["n_evaluated"],
            "budget": PRODUCTION_BUDGET,
            "seed": PRODUCTION_SEED,
        })
        print(f"{fam}: candidate={result['candidate']}, mean_error={result['mean_error']:.5f}, "
              f"n_evaluated={result['n_evaluated']}, time={elapsed:.1f}s")

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "actuator_selection", "final_actuator_sets.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "candidate", "mean_error", "n_evaluated",
                                           "budget", "seed"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nSaved", out_path)


if __name__ == "__main__":
    run()
