import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


def load_batches(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["pid"] = int(r["pid"])
            r["tercile"] = int(r["tercile"])
            r["batch"] = int(r["batch"])
            r["best_error"] = float(r["best_error"])
            rows.append(r)
    return rows


def compute_delta_p(rows):
    groups = {}
    for r in rows:
        key = (r["family"], r["pid"])
        groups.setdefault(key, []).append(r["best_error"])

    delta_rows = []
    for (family, pid), errors in groups.items():
        tercile = next(r["tercile"] for r in rows if r["family"] == family and r["pid"] == pid)
        errors = np.array(errors)
        delta_rows.append({
            "family": family,
            "pid": pid,
            "tercile": tercile,
            "delta_p": float(errors.max() - errors.min()),
            "n_batches": len(errors),
        })
    return delta_rows


def compute_delta_num(delta_rows):
    strata = {}
    for r in delta_rows:
        key = (r["family"], r["tercile"])
        strata.setdefault(key, []).append(r["delta_p"])

    summary = []
    for (family, tercile), deltas in sorted(strata.items()):
        deltas = np.array(deltas)
        summary.append({
            "family": family,
            "tercile": tercile,
            "n": len(deltas),
            "delta_num_q95": float(np.quantile(deltas, 0.95)),
            "median": float(np.median(deltas)),
            "q25": float(np.quantile(deltas, 0.25)),
            "q75": float(np.quantile(deltas, 0.75)),
            "min": float(deltas.min()),
            "max": float(deltas.max()),
            "all_values": ";".join(f"{d:.6e}" for d in sorted(deltas)),
        })
    return summary


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    batches_path = os.path.join(base, "..", "results", "calibration_batches.csv")
    rows = load_batches(batches_path)

    delta_rows = compute_delta_p(rows)
    with open(os.path.join(base, "..", "results", "calibration_delta_p.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "pid", "tercile", "delta_p", "n_batches"])
        w.writeheader()
        for r in delta_rows:
            w.writerow(r)

    summary = compute_delta_num(delta_rows)
    with open(os.path.join(base, "..", "results", "calibration_delta_num.csv"), "w", newline="") as f:
        fields = ["family", "tercile", "n", "delta_num_q95", "median", "q25", "q75", "min", "max",
                   "all_values"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary:
            w.writerow(r)

    print(f"{'family':<22}{'tercile':>8}{'n':>4}{'delta_num_q95':>16}{'median':>13}{'max':>13}")
    for r in summary:
        print(f"{r['family']:<22}{r['tercile']:>8}{r['n']:>4}{r['delta_num_q95']:>16.3e}"
              f"{r['median']:>13.3e}{r['max']:>13.3e}")
