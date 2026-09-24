import sys
import os
import time
import pickle
import itertools
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

import numpy as np

import canonical as can
import actuator as act
from test_canonical import outlet_order_for

TIME_BUDGET = 250


def checkpoint_path(family):
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"rank_{family}.pkl")


def run(family, n_z_samples=8, K=4, beta=np.log(2.0)):
    families, _ = can.build_all_families()
    G = families[family]
    outlet_order = outlet_order_for(G)
    eligible = act.eligible_actuator_edges(G)
    N = len(outlet_order)
    B = act.jac.zero_sum_basis(N)
    z_samples = act.corner_z_samples(K, beta, n_z_samples)

    path = checkpoint_path(family)
    if os.path.exists(path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        scored, resume_at = state["scored"], state["resume_at"]
    else:
        scored, resume_at = [], 0

    all_combos = list(itertools.combinations(eligible, K))
    total = len(all_combos)

    t0 = time.time()
    i = resume_at
    while i < total:
        candidate = list(all_combos[i])
        score = act.score_placement(G, candidate, outlet_order, B, z_samples)
        scored.append((score, candidate))
        i += 1
        if time.time() - t0 > TIME_BUDGET:
            break

    with open(path, "wb") as f:
        pickle.dump({"scored": scored, "resume_at": i}, f)

    done = i >= total
    print(f"{family}: {i}/{total} scored, done={done}, elapsed={time.time()-t0:.1f}s")
    return done


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("family")
    args = parser.parse_args()
    run(args.family)
