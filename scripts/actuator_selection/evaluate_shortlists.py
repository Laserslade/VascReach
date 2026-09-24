import sys
import os
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

import numpy as np

import canonical as can
import actuator as act
from test_canonical import outlet_order_for


def load_scored(family):
    path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "cache", f"rank_{family}.pkl")
    with open(path, "rb") as f:
        state = pickle.load(f)
    scored = state["scored"]
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def evaluate_family(family, G, outlet_order, screening, n_top, beta=np.log(2.0)):
    scored = load_scored(family)
    top = scored[:n_top]
    evaluated = []
    for score, candidate in top:
        mean_error = act.evaluate_candidate(G, candidate, outlet_order, screening, beta)
        evaluated.append({"score": score, "candidate": candidate, "mean_error": mean_error})
    best = min(evaluated, key=lambda r: r["mean_error"])

    n = len(scored)
    poor_idx = min(int(0.9 * n), n - 1)
    poor_score, poor_candidate = scored[poor_idx]
    poor_mean_error = act.evaluate_candidate(G, poor_candidate, outlet_order, screening, beta)

    return {
        "n_candidates": n,
        "best": best,
        "poor": {"score": poor_score, "candidate": poor_candidate, "mean_error": poor_mean_error},
    }


if __name__ == "__main__":
    families, _ = can.build_all_families()
    screening = act.frozen_screening_demand_set()

    for n_top in [5]:
        print(f"--- shortlist size {n_top} ---")
        for name, G in families.items():
            outlet_order = outlet_order_for(G)
            r = evaluate_family(name, G, outlet_order, screening, n_top)
            poor_wins = r["poor"]["mean_error"] < r["best"]["mean_error"]
            print(f"{name}: best_error={r['best']['mean_error']:.4e}, "
                  f"poor_error={r['poor']['mean_error']:.4e}, poor_wins={poor_wins}")
