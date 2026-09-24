import sys
import os
import csv
import time
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

import numpy as np

import topology_gen as tg
import canonical as can
import actuator as act
from test_canonical import outlet_order_for

BUDGETS = [10, 25, 50, 100]
SEEDS = [201, 202]
TIME_BUDGET = 200
REGIME_ORDER = ["tree", "loopy_hierarchical", "lattice", "random_spatial"]


def state_path():
    d = os.path.join(os.path.dirname(__file__), "..", "..", "results", "cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "phase2_budget_audit_state.pkl")


def load_state():
    p = state_path()
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return {"done_tasks": {}, "eval_cache": {}}


def save_state(state):
    with open(state_path(), "wb") as f:
        pickle.dump(state, f)


def load_subset():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase2", "phase2_budget_audit_subset.csv")
    return list(csv.DictReader(open(path)))


def net_id_to_graph(net_id, target_volume):
    regime, idx_str = net_id.rsplit("_", 1)
    idx = int(idx_str)
    seed, count = 0, 0
    while True:
        seed += 1
        full_seed = seed + 100000 * REGIME_ORDER.index(regime)
        G, meta = tg.generate_network(regime, full_seed, target_volume)
        if G is None:
            continue
        if count == idx:
            return G, meta
        count += 1


def run():
    subset = load_subset()
    _, target_volume = can.build_all_families()
    screening = act.frozen_screening_demand_set()
    state = load_state()

    tasks = [(r["net_id"], B, seed) for r in subset for B in BUDGETS for seed in SEEDS]
    t0 = time.time()

    for net_id, B, seed in tasks:
        key = (net_id, B, seed)
        if key in state["done_tasks"]:
            continue

        G, meta = net_id_to_graph(net_id, target_volume)
        outlet_order = outlet_order_for(G)
        cache = state["eval_cache"].setdefault(net_id, {})

        result = act.select_placement_nonlinear(
            G, outlet_order, screening, budget=B, seed=seed, cache=cache
        )
        state["done_tasks"][key] = {
            "best_error": result["mean_error"],
            "n_evaluated": result["n_evaluated"],
        }
        save_state(state)
        print(f"  {net_id} B={B} seed={seed}: err={result['mean_error']:.4f} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)

        if time.time() - t0 > TIME_BUDGET:
            print(f"Checkpoint: {len(state['done_tasks'])}/{len(tasks)} tasks done, "
                  f"elapsed={time.time()-t0:.1f}s")
            return False

    save_state(state)
    print(f"All {len(tasks)} tasks done.")
    return True


if __name__ == "__main__":
    done = run()
    print("COMPLETE" if done else "RERUN TO CONTINUE")
