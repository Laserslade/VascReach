import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1a"))

import numpy as np

import demand as dem
from run_phase1a_grid import NETWORK_SEED_OFFSET, N_DIRECTIONS


def load_grid(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["rho"] = float(r["rho"])
            r["best_error"] = float(r["best_error"])
            r["I_001"] = int(r["I_001"])
            r["direction_id"] = int(r["direction_id"])
            rows.append(r)
    return rows


def morphology_accessibility(rows):
    by_morph = {}
    for r in rows:
        key = (r["pattern_family"], round(r["rho"], 3))
        by_morph.setdefault(key, []).append(r["I_001"])

    families = sorted(set(k[0] for k in by_morph))
    rhos = sorted(set(k[1] for k in by_morph))
    out = []
    for fam in families:
        for rho in rhos:
            vals = by_morph.get((fam, rho), [])
            if vals:
                out.append({"pattern_family": fam, "rho": rho, "A": np.mean(vals), "n": len(vals)})
    return out


def multifocal_geometry_table(rows):
    positions = dem.outlet_positions_uniform(8)
    delta_x = dem.median_outlet_spacing(positions)

    net_seeds = {}
    for r in rows:
        key = (r["family"], r["variant"])
        if key not in net_seeds:
            net_seeds[key] = NETWORK_SEED_OFFSET[r["family"]] + int(r["variant"]) * 1000

    direction_cache = {}
    for key, seed in net_seeds.items():
        direction_cache[key] = dem.generate_demand_library(n_outlets=8, n_directions=N_DIRECTIONS, seed=seed)

    representative_rho = 0.2
    out = []
    seen = set()
    for r in rows:
        if r["pattern_family"] != "multifocal":
            continue
        if abs(r["rho"] - representative_rho) > 1e-6:
            continue
        key = (r["family"], r["variant"])
        direction = direction_cache[key][r["direction_id"]]
        if len(direction.centers) < 2:
            continue

        ray_key = (r["family"], r["variant"], r["direction_id"])
        if ray_key in seen:
            continue
        seen.add(ray_key)

        separation = abs(direction.centers[0] - direction.centers[1]) / delta_x
        width = float(direction.sigmas[0]) / delta_x
        out.append({
            "family": r["family"], "variant": r["variant"], "direction_id": r["direction_id"],
            "separation_over_dx": separation, "width_over_dx": width,
            "E_at_rho0.2": r["best_error"], "I_at_rho0.2": r["I_001"],
        })
    return out


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    grid_path = os.path.join(base, "..", "..", "results", "phase1a", "phase1a_dense_grid.csv")
    rows = load_grid(grid_path)

    morph = morphology_accessibility(rows)
    with open(os.path.join(base, "..", "..", "results", "phase1c", "phase1c_morphology.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pattern_family", "rho", "A", "n"])
        w.writeheader()
        for r in morph:
            w.writerow(r)

    multi = multifocal_geometry_table(rows)
    with open(os.path.join(base, "..", "..", "results", "phase1c", "phase1c_multifocal_geometry.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "variant", "direction_id", "separation_over_dx",
                                           "width_over_dx", "E_at_rho0.2", "I_at_rho0.2"])
        w.writeheader()
        for r in multi:
            w.writerow(r)

    print(f"Morphology rows: {len(morph)}, multifocal geometry rows: {len(multi)}")

    seps = np.array([r["separation_over_dx"] for r in multi])
    errs = np.array([r["E_at_rho0.2"] for r in multi])
    widths = np.array([r["width_over_dx"] for r in multi])
    if len(seps) > 2:
        rho_sep = np.corrcoef(seps, errs)[0, 1]
        rho_width = np.corrcoef(widths, errs)[0, 1]
        print(f"corr(separation, E* at rho=0.2) = {rho_sep:.3f}")
        print(f"corr(width, E* at rho=0.2) = {rho_width:.3f}")
