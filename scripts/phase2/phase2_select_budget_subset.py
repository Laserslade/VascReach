import sys
import os
import csv

import numpy as np

REGIMES = ["tree", "loopy_hierarchical", "lattice", "random_spatial"]


def load_ensemble(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["cycle_rank"] = int(r["cycle_rank"])
    return rows


def stratified_subset(rows, n_per_regime=3):
    chosen = []
    for regime in REGIMES:
        regime_rows = [r for r in rows if r["regime"] == regime]
        regime_rows.sort(key=lambda r: r["cycle_rank"])
        n = len(regime_rows)
        if n <= n_per_regime:
            chosen.extend(regime_rows)
            continue
        idxs = np.linspace(0, n - 1, n_per_regime).round().astype(int)
        idxs = sorted(set(idxs))
        while len(idxs) < n_per_regime:
            idxs.append(min(idxs[-1] + 1, n - 1))
        chosen.extend(regime_rows[i] for i in idxs)
    return chosen


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    rows = load_ensemble(os.path.join(base, "..", "..", "results", "phase2", "phase2_ensemble_100.csv"))
    subset = stratified_subset(rows)

    out_path = os.path.join(base, "..", "..", "results", "phase2", "phase2_budget_audit_subset.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["net_id", "regime", "seed", "cycle_rank"])
        w.writeheader()
        for r in subset:
            w.writerow({"net_id": r["net_id"], "regime": r["regime"],
                        "seed": r["seed"], "cycle_rank": r["cycle_rank"]})

    print(f"Selected {len(subset)} networks for the budget audit:")
    for r in subset:
        print(f"  {r['net_id']} (cycle_rank={r['cycle_rank']})")
