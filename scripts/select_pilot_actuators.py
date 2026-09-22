import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import numpy as np

import canonical as can
import actuator as act
from test_canonical import outlet_order_for

PRODUCTION_BUDGET = 100
PRODUCTION_SEED = 424242

FROZEN_VARIANT0_SETS = {
    "symmetric_tree": [("in", "in1"), ("in0", "in00"), ("in1", "in11"), ("in10", "out4")],
    "murray_tree": [("in", "in1"), ("in0", "in00"), ("in1", "in11"), ("in10", "out4")],
    "looped_hierarchical": [("in", "in0"), ("in0", "in01"), ("in1", "in10"), ("in1", "in11")],
    "grid_lattice": [("in", "mid11"), ("in", "mid8"), ("mid4", "in"), ("out6", "mid14")],
}


def run():
    pilots, tv = can.build_pilot_networks()
    screening = act.frozen_screening_demand_set()

    rows = []
    for family in can.FAMILY_BUILDERS:
        for variant in range(3):
            G = pilots[(family, variant)]
            outlet_order = outlet_order_for(G)

            if variant == 0:
                candidate = FROZEN_VARIANT0_SETS[family]
                mean_error = act.evaluate_candidate(G, candidate, outlet_order, screening, np.log(2.0))
                n_evaluated = "reused"
                elapsed = 0.0
            else:
                t0 = time.time()
                result = act.select_placement_nonlinear(
                    G, outlet_order, screening, budget=PRODUCTION_BUDGET, seed=PRODUCTION_SEED
                )
                candidate = result["candidate"]
                mean_error = result["mean_error"]
                n_evaluated = result["n_evaluated"]
                elapsed = time.time() - t0

            rows.append({
                "family": family, "variant": variant, "candidate": str(candidate),
                "mean_error": mean_error, "n_evaluated": n_evaluated, "elapsed": elapsed,
            })
            print(f"{family} variant {variant}: candidate={candidate}, "
                  f"mean_error={mean_error:.5f}, elapsed={elapsed:.1f}s")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "pilot_actuator_sets.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "variant", "candidate", "mean_error",
                                           "n_evaluated", "elapsed"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("Saved", out_path)


if __name__ == "__main__":
    run()
