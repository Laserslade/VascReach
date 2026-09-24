import sys
import os
import csv
import time
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

import numpy as np
from scipy.optimize import differential_evolution, minimize

import canonical as can
import control as ctl
import demand as dem
from test_canonical import outlet_order_for
from run_phase1a_grid import ACTUATOR_SETS, NETWORK_SEED_OFFSET, N_DIRECTIONS

BETA = np.log(2.0)
TIME_BUDGET = 250


def state_path():
    d = os.path.join(os.path.dirname(__file__), "..", "..", "results", "cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "phase1a_verify_state.pkl")


def load_state():
    p = state_path()
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return {"done": {}}


def save_state(state):
    with open(state_path(), "wb") as f:
        pickle.dump(state, f)


def build_task_list():
    tasks = set()
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase1a", "phase1a_primary_events.csv")) as f:
        for r in csv.DictReader(f):
            tasks.add((r["family"], int(r["variant"]), int(r["direction_id"]), round(float(r["rho_j"]), 3)))
            tasks.add((r["family"], int(r["variant"]), int(r["direction_id"]), round(float(r["rho_k"]), 3)))

    with open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase1a", "phase1a_secondary_events.csv")) as f:
        for r in csv.DictReader(f):
            if r["escalate"] == "True":
                tasks.add((r["family"], int(r["variant"]), int(r["direction_id"]), round(float(r["rho_j"]), 3)))
                tasks.add((r["family"], int(r["variant"]), int(r["direction_id"]), round(float(r["rho_k"]), 3)))

    return sorted(tasks)


def run():
    tasks = build_task_list()
    print(f"Total distinct verification points: {len(tasks)}")

    pilots, tv = can.build_pilot_networks()
    state = load_state()
    b = np.full(8, 1.0 / 8)

    directions_cache = {}

    t0 = time.time()
    for task in tasks:
        if task in state["done"]:
            continue

        family, variant, direction_id, rho = task
        G = pilots[(family, variant)]
        outlet_order = outlet_order_for(G)
        candidate = ACTUATOR_SETS[(family, variant)]

        net_seed = NETWORK_SEED_OFFSET[family] + variant * 1000
        cache_key = (family, variant)
        if cache_key not in directions_cache:
            directions_cache[cache_key] = dem.generate_demand_library(
                n_outlets=8, n_directions=N_DIRECTIONS, seed=net_seed
            )
        direction = directions_cache[cache_key][direction_id]

        s = rho * direction.s_max
        target = b + s * direction.v_hat

        def obj(z, G=G, candidate=candidate, outlet_order=outlet_order, target=target):
            L, _ = ctl.objective_and_grad(z, G, candidate, outlet_order, target)
            return L

        de_res = differential_evolution(
            obj, bounds=[(-BETA, BETA)] * len(candidate), seed=42,
            tol=1e-12, maxiter=200, polish=False,
        )
        polished = minimize(
            ctl.objective_and_grad, de_res.x, args=(G, candidate, outlet_order, target),
            jac=True, method="L-BFGS-B", bounds=[(-BETA, BETA)] * len(candidate),
        )
        de_error = float(np.sqrt(max(polished.fun, 0.0)))

        state["done"][task] = de_error

        if time.time() - t0 > TIME_BUDGET:
            save_state(state)
            print(f"Checkpoint: {len(state['done'])}/{len(tasks)} verified, elapsed={time.time()-t0:.1f}s")
            return False

    save_state(state)
    print(f"All {len(tasks)} verification points complete.")
    return True


if __name__ == "__main__":
    done = run()
    print("COMPLETE" if done else "RERUN TO CONTINUE")
