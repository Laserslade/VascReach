import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1a"))

import numpy as np

import canonical as can
import control as ctl
import demand as dem
from test_canonical import outlet_order_for
from run_phase1a_grid import ACTUATOR_SETS, NETWORK_SEED_OFFSET, N_DIRECTIONS

N_STARTS = 8
RHO_SUBSET = np.round(np.arange(0.0, 1.0, 0.1), 3)

BETAS = {
    "low": np.log(1.25),
    "frozen": np.log(2.0),
    "high": np.log(4.0),
}


def run():
    pilots, tv = can.build_pilot_networks()
    b = np.full(8, 1.0 / 8)

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase1c", "phase1c_authority_sweep.csv")
    f = open(out_path, "w", newline="")
    fields = ["family", "beta_label", "beta_value", "direction_id", "pattern_family",
              "rho", "s", "best_error", "I_001"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    t0 = time.time()
    for family in can.FAMILY_BUILDERS:
        G = pilots[(family, 0)]
        outlet_order = outlet_order_for(G)
        candidate = ACTUATOR_SETS[(family, 0)]
        net_seed = NETWORK_SEED_OFFSET[family]
        directions = dem.generate_demand_library(n_outlets=8, n_directions=N_DIRECTIONS, seed=net_seed)

        for beta_label, beta_val in BETAS.items():
            for direction in directions:
                for j, rho in enumerate(RHO_SUBSET):
                    s = rho * direction.s_max
                    target = b + s * direction.v_hat
                    seed = net_seed + direction.direction_id * 1000 + j + 500000
                    result = ctl.solve_inverse(
                        G, candidate, outlet_order, target, beta_val,
                        n_starts=N_STARTS, seed=seed,
                    )
                    writer.writerow({
                        "family": family, "beta_label": beta_label, "beta_value": beta_val,
                        "direction_id": direction.direction_id,
                        "pattern_family": direction.pattern_family,
                        "rho": rho, "s": s, "best_error": result["best_error"],
                        "I_001": int(result["best_error"] <= 0.01),
                    })
            print(f"{family} beta={beta_label} done, elapsed={time.time()-t0:.1f}s")
            f.flush()

    f.close()
    print("Saved", out_path)


if __name__ == "__main__":
    run()
