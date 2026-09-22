import sys
import os
import time
import pickle
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import numpy as np

import canonical as can
import actuator as act
from test_canonical import outlet_order_for

TIME_BUDGET = 250
BUDGETS = [25, 50, 100]
SEEDS = [101, 102, 103, 104, 105]
FAMILIES = ["symmetric_tree", "murray_tree", "looped_hierarchical", "grid_lattice"]


def state_path():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "cache")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "stability_audit_state.pkl")


def load_state():
    path = state_path()
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {"done_tasks": {}, "eval_cache": {}}


def save_state(state):
    with open(state_path(), "wb") as f:
        pickle.dump(state, f)


def run():
    families, _ = can.build_all_families()
    screening = act.frozen_screening_demand_set()
    state = load_state()

    tasks = [(fam, B, seed) for fam in FAMILIES for B in BUDGETS for seed in SEEDS]
    t0 = time.time()

    for fam, B, seed in tasks:
        key = (fam, B, seed)
        if key in state["done_tasks"]:
            continue

        G = families[fam]
        outlet_order = outlet_order_for(G)
        cache = state["eval_cache"].setdefault(fam, {})

        result = act.select_placement_nonlinear(
            G, outlet_order, screening, budget=B, seed=seed, cache=cache
        )
        state["done_tasks"][key] = {
            "best_error": result["mean_error"],
            "candidate": result["candidate"],
            "n_evaluated": result["n_evaluated"],
        }

        if time.time() - t0 > TIME_BUDGET:
            save_state(state)
            print(f"Checkpoint: {len(state['done_tasks'])}/{len(tasks)} tasks done, "
                  f"elapsed={time.time()-t0:.1f}s")
            return False

    save_state(state)
    print(f"All {len(tasks)} tasks done.")
    return True


if __name__ == "__main__":
    done = run()
    if done:
        print("Stability audit complete.")
    else:
        print("Rerun to continue.")
