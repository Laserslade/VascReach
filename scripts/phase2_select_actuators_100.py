import sys
import os
import csv
import time
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import numpy as np

import topology_gen as tg
import canonical as can
import actuator as act
from test_canonical import outlet_order_for
from phase2_budget_audit import net_id_to_graph, REGIME_ORDER

BUDGET = 25
PRODUCTION_SEED = 424242
TIME_BUDGET = 250


def state_path():
    d = os.path.join(os.path.dirname(__file__), "..", "results", "cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "phase2_actuator_selection_state.pkl")


def load_state():
    p = state_path()
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return {"done": {}}


def save_state(state):
    with open(state_path(), "wb") as f:
        pickle.dump(state, f)


def load_ensemble():
    path = os.path.join(os.path.dirname(__file__), "..", "results", "phase2_ensemble_100.csv")
    return list(csv.DictReader(open(path)))


def run():
    ensemble = load_ensemble()
    _, target_volume = can.build_all_families()
    screening = act.frozen_screening_demand_set()
    state = load_state()

    t0 = time.time()
    for r in ensemble:
        net_id = r["net_id"]
        if net_id in state["done"]:
            continue

        G, meta = net_id_to_graph(net_id, target_volume)
        outlet_order = outlet_order_for(G)
        result = act.select_placement_nonlinear(
            G, outlet_order, screening, budget=BUDGET, seed=PRODUCTION_SEED
        )
        state["done"][net_id] = {"candidate": result["candidate"], "mean_error": result["mean_error"]}
        save_state(state)
        print(f"  {net_id}: err={result['mean_error']:.4f} elapsed={time.time()-t0:.1f}s", flush=True)

        if time.time() - t0 > TIME_BUDGET:
            print(f"Checkpoint: {len(state['done'])}/{len(ensemble)} done")
            return False

    print(f"All {len(ensemble)} actuator selections done.")
    return True


if __name__ == "__main__":
    done = run()
    print("COMPLETE" if done else "RERUN TO CONTINUE")
