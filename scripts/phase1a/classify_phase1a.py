import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

TERCILE_EDGES = [(0.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0)]


def tercile_of(rho):
    for i, (lo, hi) in enumerate(TERCILE_EDGES):
        if lo <= rho < hi or (i == 2 and rho <= 1.0):
            return i
    return 2


def load_delta_num():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "calibration", "calibration_delta_num.csv")
    table = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            table[(r["family"], int(r["tercile"]))] = float(r["delta_num_q95"])
    return table


def load_grid():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase1a", "phase1a_dense_grid.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["rho"] = float(r["rho"])
            r["s"] = float(r["s"])
            r["s_max"] = float(r["s_max"])
            r["best_error"] = float(r["best_error"])
            r["I_001"] = int(r["I_001"])
            r["direction_id"] = int(r["direction_id"])
            r["variant"] = int(r["variant"])
            rows.append(r)
    return rows


def classify():
    delta_num = load_delta_num()
    rows = load_grid()

    groups = {}
    for r in rows:
        key = (r["family"], r["variant"], r["direction_id"])
        groups.setdefault(key, []).append(r)

    primary_events = []
    secondary_events = []
    ray_summaries = []

    for key, group in groups.items():
        family, variant, direction_id = key
        group.sort(key=lambda r: r["rho"])
        I = [r["I_001"] for r in group]
        E = [r["best_error"] for r in group]
        rhos = [r["rho"] for r in group]

        first_zero_idx = next((i for i, v in enumerate(I) if v == 0), None)
        reentry = False
        reentry_points = []
        if first_zero_idx is not None:
            for k in range(first_zero_idx + 1, len(I)):
                if I[k] == 1:
                    reentry = True
                    reentry_points.append((first_zero_idx, k))

        if reentry:
            for j, k in reentry_points:
                primary_events.append({
                    "family": family, "variant": variant, "direction_id": direction_id,
                    "rho_j": rhos[j], "E_j": E[j], "rho_k": rhos[k], "E_k": E[k],
                })

        largest_reversals = []
        for j in range(len(group) - 1):
            reversal = E[j] - E[j + 1]
            q = tercile_of(rhos[j])
            threshold = delta_num.get((family, q), 0.0)
            if reversal > threshold:
                near_boundary = abs(E[j] - 0.01) < 0.002 or abs(E[j + 1] - 0.01) < 0.002
                largest_reversals.append({
                    "family": family, "variant": variant, "direction_id": direction_id,
                    "rho_j": rhos[j], "rho_k": rhos[j + 1], "reversal": reversal,
                    "threshold": threshold, "near_boundary": near_boundary,
                })

        largest_reversals.sort(key=lambda e: -e["reversal"])
        for idx, e in enumerate(largest_reversals):
            e["escalate"] = (idx == 0) or (e["near_boundary"] and e["reversal"] > 1e-4)
            secondary_events.append(e)

        ray_summaries.append({
            "family": family, "variant": variant, "direction_id": direction_id,
            "n_points": len(group), "reentry": reentry, "n_reversals_flagged": len(largest_reversals),
            "final_I": I[-1], "all_accessible": all(v == 1 for v in I),
        })

    return primary_events, secondary_events, ray_summaries


if __name__ == "__main__":
    primary, secondary, rays = classify()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase1a")

    with open(os.path.join(out_dir, "phase1a_primary_events.csv"), "w", newline="") as f:
        fields = ["family", "variant", "direction_id", "rho_j", "E_j", "rho_k", "E_k"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in primary:
            w.writerow(e)

    with open(os.path.join(out_dir, "phase1a_secondary_events.csv"), "w", newline="") as f:
        fields = ["family", "variant", "direction_id", "rho_j", "rho_k", "reversal",
                   "threshold", "near_boundary", "escalate"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in secondary:
            w.writerow(e)

    with open(os.path.join(out_dir, "phase1a_ray_summary.csv"), "w", newline="") as f:
        fields = ["family", "variant", "direction_id", "n_points", "reentry",
                   "n_reversals_flagged", "final_I", "all_accessible"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rays:
            w.writerow(r)

    n_rays = len(rays)
    n_reentry_rays = sum(1 for r in rays if r["reentry"])
    n_escalate = sum(1 for e in secondary if e["escalate"])

    print(f"Total rays: {n_rays}")
    print(f"Rays with primary (0->1) re-entry: {n_reentry_rays}")
    print(f"Total primary re-entry events: {len(primary)}")
    print(f"Total secondary reversal events flagged: {len(secondary)}")
    print(f"Secondary events escalated for verification: {n_escalate}")
