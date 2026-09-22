import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


def load_rays(path):
    rays = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["family"], r["variant"], r["direction_id"])
            rays.setdefault(key, {"s": [], "E": [], "s_max": float(r["s_max"])})
            rays[key]["s"].append(float(r["s"]))
            rays[key]["E"].append(float(r["best_error"]))
    for key, d in rays.items():
        order = np.argsort(d["s"])
        d["s"] = np.array(d["s"])[order]
        d["E"] = np.array(d["E"])[order]
    return rays


def interp_E_on_grid(rays, s_grid, epsilon=0.01):
    per_ray_I = {}
    for key, d in rays.items():
        s_valid_max = d["s"].max()
        E_interp = np.interp(s_grid, d["s"], d["E"])
        valid = s_grid <= s_valid_max
        I = np.where(valid, (E_interp <= epsilon).astype(float), np.nan)
        per_ray_I[key] = {"I": I, "valid": valid, "E_interp": E_interp}
    families = sorted(set(k[0] for k in rays))
    return per_ray_I, families


def aggregate(per_ray_I, keys_filter, s_grid):
    keys = [k for k in per_ray_I if keys_filter(k)]
    I_stack = np.array([per_ray_I[k]["I"] for k in keys])
    valid_stack = np.array([per_ray_I[k]["valid"] for k in keys])
    n_valid = valid_stack.sum(axis=0)
    with np.errstate(invalid="ignore"):
        A = np.nansum(np.where(valid_stack, I_stack, 0), axis=0) / np.maximum(n_valid, 1)
    A = np.where(n_valid > 0, A, np.nan)
    return A, n_valid


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    grid_path = os.path.join(base, "..", "results", "phase1a_dense_grid.csv")
    rays = load_rays(grid_path)

    s_grid = np.linspace(0.0, 0.5, 26)
    per_ray_I, families = interp_E_on_grid(rays, s_grid)

    out_dir = os.path.join(base, "..", "results")

    with open(os.path.join(out_dir, "phase1b_family_accessibility.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "family"] + [f"s{i}" for i in range(len(s_grid))])
        for fam in families:
            A, n_valid = aggregate(per_ray_I, lambda k, fam=fam: k[0] == fam, s_grid)
            w.writerow(["A", fam] + list(A))
            w.writerow(["n_valid", fam] + list(n_valid))
            print(fam, "A(s):", np.round(A, 3))
            print(fam, "n_valid(s):", n_valid.astype(int))
        w.writerow(["s_grid", ""] + list(s_grid))

    networks = sorted(set((k[0], k[1]) for k in per_ray_I))
    with open(os.path.join(out_dir, "phase1b_network_accessibility.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "family", "variant"] + [f"s{i}" for i in range(len(s_grid))])
        for fam, var in networks:
            A, n_valid = aggregate(per_ray_I, lambda k, fam=fam, var=var: k[0] == fam and k[1] == var, s_grid)
            w.writerow(["A", fam, var] + list(A))
            w.writerow(["n_valid", fam, var] + list(n_valid))
        w.writerow(["s_grid", "", ""] + list(s_grid))

    print("\nSaved phase1b_family_accessibility.csv and phase1b_network_accessibility.csv")
