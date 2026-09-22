import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import numpy as np

import canonical as can
import control as ctl
import calibration as cal
from test_canonical import outlet_order_for

BETA = np.log(2.0)
N_BATCHES = 5
N_PER_STRATUM = 20
N_STARTS = 8

FROZEN_ACTUATOR_SETS = {
    "symmetric_tree": [("in", "in1"), ("in0", "in00"), ("in1", "in11"), ("in10", "out4")],
    "murray_tree": [("in", "in1"), ("in0", "in00"), ("in1", "in11"), ("in10", "out4")],
    "looped_hierarchical": [("in", "in0"), ("in0", "in01"), ("in1", "in10"), ("in1", "in11")],
    "grid_lattice": [("in", "mid11"), ("in", "mid8"), ("mid4", "in"), ("out6", "mid14")],
}

FAMILY_SEED_OFFSET = {
    "symmetric_tree": 0,
    "murray_tree": 1000,
    "looped_hierarchical": 2000,
    "grid_lattice": 3000,
}


def run():
    families, _ = can.build_all_families()
    problems = cal.generate_calibration_problems(n_per_stratum=N_PER_STRATUM)
    b = np.full(8, 1.0 / 8)

    batch_rows = []
    t0 = time.time()

    for fam_name, G in families.items():
        outlet_order = outlet_order_for(G)
        candidate = FROZEN_ACTUATOR_SETS[fam_name]

        for prob in problems:
            target = b + prob["s"] * prob["v_hat"]
            base_seed = 900000 + FAMILY_SEED_OFFSET[fam_name] + prob["pid"] * 7

            for r in range(N_BATCHES):
                batch_seed = base_seed + r
                result = ctl.solve_inverse(
                    G, candidate, outlet_order, target, BETA,
                    n_starts=N_STARTS, seed=batch_seed,
                )
                batch_rows.append({
                    "family": fam_name,
                    "pid": prob["pid"],
                    "tercile": prob["tercile"],
                    "pattern_family": prob["pattern_family"],
                    "rho": prob["rho"],
                    "s": prob["s"],
                    "batch": r,
                    "batch_seed": batch_seed,
                    "best_error": result["best_error"],
                    "optimizer_success": result["optimizer_success"],
                })

        print(f"{fam_name} done, elapsed={time.time()-t0:.1f}s")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "calibration_batches.csv")
    with open(out_path, "w", newline="") as f:
        fields = ["family", "pid", "tercile", "pattern_family", "rho", "s", "batch",
                   "batch_seed", "best_error", "optimizer_success"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in batch_rows:
            w.writerow(r)

    print("Saved", out_path, "total elapsed", time.time() - t0)


if __name__ == "__main__":
    run()
