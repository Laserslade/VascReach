import sys
import os
import csv
import pickle

import numpy as np

BUDGETS = [10, 25, 50, 100]


def load_results():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "cache",
                         "phase2_budget_audit_state.pkl")
    with open(path, "rb") as f:
        state = pickle.load(f)
    return state["done_tasks"]


def analyze():
    done = load_results()
    by_net_B = {}
    for (net_id, B, seed), r in done.items():
        by_net_B.setdefault((net_id, B), []).append(r["best_error"])

    networks = sorted(set(k[0] for k in by_net_B))

    rows = []
    for net_id in networks:
        for B in BUDGETS:
            errs = by_net_B[(net_id, B)]
            rows.append({"net_id": net_id, "B": B, "mean_err": np.mean(errs),
                         "seed_spread": max(errs) - min(errs), "n_seeds": len(errs)})

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase2",
                             "phase2_budget_audit_analysis.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["net_id", "B", "mean_err", "seed_spread", "n_seeds"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"{'B':>5}{'pooled mean err':>18}{'pooled std':>14}{'mean seed_spread':>18}")
    for B in BUDGETS:
        vals = [r["mean_err"] for r in rows if r["B"] == B]
        spreads = [r["seed_spread"] for r in rows if r["B"] == B]
        print(f"{B:>5}{np.mean(vals):>18.4f}{np.std(vals):>14.4f}{np.mean(spreads):>18.4f}")

    print()
    print("Per-network mean error by B (rows=networks, cols=B):")
    print(f"{'net_id':<22}" + "".join(f"{B:>10}" for B in BUDGETS))
    for net_id in networks:
        line = f"{net_id:<22}"
        for B in BUDGETS:
            v = next(r["mean_err"] for r in rows if r["net_id"] == net_id and r["B"] == B)
            line += f"{v:>10.4f}"
        print(line)


if __name__ == "__main__":
    analyze()
