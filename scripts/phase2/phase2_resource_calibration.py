import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

import numpy as np

import canonical as can
import control as ctl
import actuator as act
from test_canonical import outlet_order_for
from phase2_budget_audit import net_id_to_graph

K_VALUES = [4, 6, 8]
BETA_VALUES = {"log2": np.log(2.0), "log4": np.log(4.0)}
PLACEMENT_BUDGET = 20
PLACEMENT_SEED = 424242
N_STARTS = 8


def load_subset():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase2", "phase2_budget_audit_subset.csv")
    return [r["net_id"] for r in csv.DictReader(open(path))]


def run():
    net_ids = load_subset()
    _, target_volume = can.build_all_families()
    b = np.full(8, 1.0 / 8)

    rows = []
    t0 = time.time()
    for net_id in net_ids:
        G, meta = net_id_to_graph(net_id, target_volume)
        outlet_order = outlet_order_for(G)

        for K in K_VALUES:
            placement_result = act.select_placement_nonlinear(
                G, outlet_order, screening_demands=[b], K=K,
                budget=PLACEMENT_BUDGET, seed=PLACEMENT_SEED,
            )
            candidate = placement_result["candidate"]

            for beta_label, beta_val in BETA_VALUES.items():
                result = ctl.solve_inverse(
                    G, candidate, outlet_order, b, beta_val,
                    n_starts=N_STARTS, seed=PLACEMENT_SEED,
                )
                rows.append({
                    "net_id": net_id, "K": K, "beta_label": beta_label,
                    "beta_value": beta_val, "E0_star": result["best_error"],
                    "correctable": int(result["best_error"] <= 0.01),
                })
        print(f"{net_id} done, elapsed={time.time()-t0:.1f}s", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase2",
                             "phase2_resource_calibration.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["net_id", "K", "beta_label", "beta_value",
                                           "E0_star", "correctable"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print()
    print(f"{'K':>4}{'beta':>8}{'C(K,beta)':>12}{'mean E0*':>12}{'median E0*':>14}")
    for K in K_VALUES:
        for beta_label in BETA_VALUES:
            sub = [r for r in rows if r["K"] == K and r["beta_label"] == beta_label]
            C = np.mean([r["correctable"] for r in sub])
            mean_e = np.mean([r["E0_star"] for r in sub])
            median_e = np.median([r["E0_star"] for r in sub])
            print(f"{K:>4}{beta_label:>8}{C:>12.3f}{mean_e:>12.4f}{median_e:>14.4f}")


if __name__ == "__main__":
    run()
