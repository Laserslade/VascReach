import sys
import os
import csv

import numpy as np
from scipy.stats import spearmanr

COVERAGE_THRESHOLD = 0.90


def load_grid(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["s"] = float(r["s"])
            r["direction_id"] = int(r["direction_id"])
            r["valid"] = r["valid"] == "1"
            if r["valid"]:
                r["I_001"] = int(r["I_001"])
            rows.append(r)
    return rows


def determine_s_end(rows):
    s_vals = sorted(set(r["s"] for r in rows))
    s_end = 0.0
    for s in s_vals:
        n_valid = sum(1 for r in rows if r["s"] == s and r["valid"])
        n_total = sum(1 for r in rows if r["s"] == s)
        coverage = n_valid / n_total
        if coverage >= COVERAGE_THRESHOLD:
            s_end = s
        else:
            break
    return s_end, [s for s in s_vals if s <= s_end]


def compute_A_at_s(rows, net_id, s, max_direction_id):
    pts = [r for r in rows if r["net_id"] == net_id and r["s"] == s
           and r["direction_id"] < max_direction_id]
    valid_pts = [r for r in pts if r["valid"]]
    if len(valid_pts) == 0:
        return np.nan
    return np.mean([r["I_001"] for r in valid_pts])


def trapezoidal_aac(s_vals, A_vals):
    s_vals = np.array(s_vals)
    A_vals = np.array(A_vals)
    mask = ~np.isnan(A_vals)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapezoid(A_vals[mask], s_vals[mask]))


def compute_aac_table(rows, s_grid, network_ids):
    results = {}
    for n_dirs in [10, 20, 30]:
        results[n_dirs] = {}
        for net_id in network_ids:
            A_vals = [compute_A_at_s(rows, net_id, s, n_dirs) for s in s_grid]
            results[n_dirs][net_id] = trapezoidal_aac(s_grid, A_vals)
    return results


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    grid_path = os.path.join(base, "..", "results", "phase2_eval_30x8.csv")
    rows = load_grid(grid_path)

    s_end, s_grid = determine_s_end(rows)
    print(f"Predeclared coverage threshold: {COVERAGE_THRESHOLD}")
    print(f"s_end = {s_end}, integration grid = {s_grid}")

    network_ids = sorted(set(r["net_id"] for r in rows))
    aac = compute_aac_table(rows, s_grid, network_ids)

    out_path = os.path.join(base, "..", "results", "phase2_aac_convergence.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["net_id", "AAC_10", "AAC_20", "AAC_30"])
        w.writeheader()
        for net_id in network_ids:
            w.writerow({"net_id": net_id, "AAC_10": aac[10][net_id],
                        "AAC_20": aac[20][net_id], "AAC_30": aac[30][net_id]})

    aac30 = np.array([aac[30][n] for n in network_ids])
    print()
    for n_dirs in [10, 20]:
        aac_sub = np.array([aac[n_dirs][n] for n in network_ids])
        rho, p = spearmanr(aac_sub, aac30)
        abs_err = np.abs(aac_sub - aac30)
        median_err = np.median(abs_err)
        p95_err = np.percentile(abs_err, 95)
        print(f"AAC_{n_dirs} vs AAC_30: Spearman rho={rho:.4f} (p={p:.2e}), "
              f"median |err|={median_err:.5f}, p95 |err|={p95_err:.5f}")
