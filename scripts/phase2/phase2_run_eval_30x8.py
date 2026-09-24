import sys
import os
import csv
import time
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

import numpy as np

import canonical as can
import control as ctl
import demand as dem
from test_canonical import outlet_order_for
from phase2_budget_audit import net_id_to_graph, REGIME_ORDER

N_DIRECTIONS = 30
N_STARTS = 8
S_GRID = np.round(np.linspace(0.0, 0.25, 8), 4)
TIME_BUDGET = 250
NET_SEED_BASE = 700000


def net_id_seed(net_id):
    regime, idx_str = net_id.rsplit("_", 1)
    return NET_SEED_BASE + REGIME_ORDER.index(regime) * 10000 + int(idx_str) * 10


def state_path():
    d = os.path.join(os.path.dirname(__file__), "..", "..", "results", "cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "phase2_eval_state.pkl")


def csv_path():
    return os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase2", "phase2_eval_30x8.csv")


def load_state():
    p = state_path()
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return {"done_networks": []}


def save_state(state):
    with open(state_path(), "wb") as f:
        pickle.dump(state, f)


def load_ensemble():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase2", "phase2_ensemble_100.csv")
    return list(csv.DictReader(open(path)))


def load_actuator_sets():
    p = os.path.join(os.path.dirname(__file__), "..", "..", "results", "cache",
                      "phase2_actuator_selection_state.pkl")
    with open(p, "rb") as f:
        state = pickle.load(f)
    return state["done"]


def run():
    ensemble = load_ensemble()
    actuator_sets = load_actuator_sets()
    _, target_volume = can.build_all_families()
    state = load_state()
    b = np.full(8, 1.0 / 8)

    write_header = not os.path.exists(csv_path())
    f = open(csv_path(), "a", newline="")
    fields = ["net_id", "regime", "direction_id", "pattern_family", "rho", "s", "s_max",
              "valid", "best_error", "I_001", "optimizer_success"]
    writer = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        writer.writeheader()

    t0 = time.time()
    for r in ensemble:
        net_id = r["net_id"]
        if net_id in state["done_networks"]:
            continue

        G, meta = net_id_to_graph(net_id, target_volume)
        outlet_order = outlet_order_for(G)
        candidate = actuator_sets[net_id]["candidate"]

        net_seed = net_id_seed(net_id)
        directions = dem.generate_demand_library(n_outlets=8, n_directions=N_DIRECTIONS, seed=net_seed)

        for direction in directions:
            for j, s in enumerate(S_GRID):
                rho = float(s / direction.s_max) if direction.s_max > 0 else np.nan
                if s >= direction.s_max:
                    writer.writerow({
                        "net_id": net_id, "regime": r["regime"],
                        "direction_id": direction.direction_id,
                        "pattern_family": direction.pattern_family,
                        "rho": rho, "s": s, "s_max": direction.s_max,
                        "valid": 0, "best_error": "", "I_001": "", "optimizer_success": "",
                    })
                    continue
                target = b + s * direction.v_hat
                seed = net_seed + direction.direction_id * 1000 + j
                result = ctl.solve_inverse(
                    G, candidate, outlet_order, target, np.log(2.0),
                    n_starts=N_STARTS, seed=seed,
                )
                writer.writerow({
                    "net_id": net_id, "regime": r["regime"],
                    "direction_id": direction.direction_id,
                    "pattern_family": direction.pattern_family,
                    "rho": rho, "s": s, "s_max": direction.s_max,
                    "valid": 1, "best_error": result["best_error"],
                    "I_001": int(result["best_error"] <= 0.01),
                    "optimizer_success": result["optimizer_success"],
                })

        state["done_networks"].append(net_id)
        save_state(state)
        f.flush()
        elapsed = time.time() - t0
        print(f"{net_id} done, elapsed={elapsed:.1f}s", flush=True)

        if elapsed > TIME_BUDGET:
            f.close()
            print(f"Checkpoint: {len(state['done_networks'])}/100 networks done")
            return False

    f.close()
    print("All 100 networks done.")
    return True


if __name__ == "__main__":
    done = run()
    print("COMPLETE" if done else "RERUN TO CONTINUE")
