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

BETA = np.log(2.0)
N_DIRECTIONS = 20
N_STARTS = 8
RHO_GRID = np.round(np.arange(0.0, 1.0, 0.025), 3)
TIME_BUDGET = 250

ACTUATOR_SETS = {
    ("symmetric_tree", 0): [("in", "in1"), ("in0", "in00"), ("in1", "in11"), ("in10", "out4")],
    ("symmetric_tree", 1): [("in", "in0"), ("in0", "in00"), ("in1", "in11"), ("in11", "out6")],
    ("symmetric_tree", 2): [("in", "in1"), ("in0", "in00"), ("in1", "in11"), ("in10", "out4")],
    ("murray_tree", 0): [("in", "in1"), ("in0", "in00"), ("in1", "in11"), ("in10", "out4")],
    ("murray_tree", 1): [("in", "in1"), ("in0", "in00"), ("in1", "in11"), ("in10", "out4")],
    ("murray_tree", 2): [("in", "in1"), ("in0", "in00"), ("in1", "in11"), ("in10", "out4")],
    ("looped_hierarchical", 0): [("in", "in0"), ("in0", "in01"), ("in1", "in10"), ("in1", "in11")],
    ("looped_hierarchical", 1): [("in", "in0"), ("in0", "in01"), ("in1", "in10"), ("in1", "in11")],
    ("looped_hierarchical", 2): [("in", "in0"), ("in0", "in01"), ("in1", "in10"), ("in1", "in11")],
    ("grid_lattice", 0): [("in", "mid11"), ("in", "mid8"), ("mid4", "in"), ("out6", "mid14")],
    ("grid_lattice", 1): [("in", "mid11"), ("in", "mid8"), ("mid4", "in"), ("out6", "mid14")],
    ("grid_lattice", 2): [("in", "mid11"), ("in", "mid8"), ("mid4", "in"), ("out6", "mid14")],
}

NETWORK_SEED_OFFSET = {
    "symmetric_tree": 10000,
    "murray_tree": 20000,
    "looped_hierarchical": 30000,
    "grid_lattice": 40000,
}


def state_path():
    d = os.path.join(os.path.dirname(__file__), "..", "..", "results", "cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "phase1a_state.pkl")


def csv_path():
    return os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase1a", "phase1a_dense_grid.csv")


def load_state():
    p = state_path()
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return {"done_networks": []}


def save_state(state):
    with open(state_path(), "wb") as f:
        pickle.dump(state, f)


def run():
    pilots, tv = can.build_pilot_networks()
    state = load_state()

    write_header = not os.path.exists(csv_path())
    f = open(csv_path(), "a", newline="")
    fields = ["family", "variant", "pattern_family", "direction_id", "rho", "s", "s_max",
              "best_error", "I_001", "optimizer_success", "control_effort"]
    writer = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        writer.writeheader()

    t0 = time.time()
    b = np.full(8, 1.0 / 8)

    for (family, variant), G in pilots.items():
        key = f"{family}_{variant}"
        if key in state["done_networks"]:
            continue

        outlet_order = outlet_order_for(G)
        candidate = ACTUATOR_SETS[(family, variant)]
        net_seed = NETWORK_SEED_OFFSET[family] + variant * 1000
        directions = dem.generate_demand_library(n_outlets=8, n_directions=N_DIRECTIONS, seed=net_seed)

        for direction in directions:
            for j, rho in enumerate(RHO_GRID):
                s = rho * direction.s_max
                target = b + s * direction.v_hat
                solve_seed = net_seed + direction.direction_id * 1000 + j
                result = ctl.solve_inverse(
                    G, candidate, outlet_order, target, BETA,
                    n_starts=N_STARTS, seed=solve_seed,
                )
                writer.writerow({
                    "family": family, "variant": variant,
                    "pattern_family": direction.pattern_family,
                    "direction_id": direction.direction_id,
                    "rho": rho, "s": s, "s_max": direction.s_max,
                    "best_error": result["best_error"],
                    "I_001": int(result["best_error"] <= 0.01),
                    "optimizer_success": result["optimizer_success"],
                    "control_effort": result["control_effort"],
                })

        state["done_networks"].append(key)
        save_state(state)
        f.flush()
        elapsed = time.time() - t0
        print(f"{key} done, elapsed={elapsed:.1f}s")

        if elapsed > TIME_BUDGET:
            f.close()
            print(f"Checkpoint: {len(state['done_networks'])}/12 networks done")
            return False

    save_state(state)
    f.close()
    print("All 12 networks done.")
    return True


if __name__ == "__main__":
    done = run()
    print("COMPLETE" if done else "RERUN TO CONTINUE")
