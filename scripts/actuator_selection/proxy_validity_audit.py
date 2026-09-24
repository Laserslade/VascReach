import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

import numpy as np
from scipy.stats import spearmanr

import canonical as can
import actuator as act
import jacobian as jac
from test_canonical import outlet_order_for
from evaluate_shortlists import load_scored

BETA = np.log(2.0)
N_PER_STRATUM = 20
N_STRATA = 5


def stratified_sample(scored, seed):
    n = len(scored)
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, n, N_STRATA + 1).astype(int)
    sampled = []
    for k in range(N_STRATA):
        lo, hi = edges[k], edges[k + 1]
        idx_pool = np.arange(lo, hi)
        take = min(N_PER_STRATUM, len(idx_pool))
        chosen = rng.choice(idx_pool, size=take, replace=False)
        for i in chosen:
            sampled.append((k, scored[i]))
    return sampled


def rescore_16(G, candidate, outlet_order, B):
    z_samples = act.corner_z_samples(len(candidate), BETA, 16)
    return act.score_placement(G, candidate, outlet_order, B, z_samples)


def audit_family(family_name, seed):
    families, _ = can.build_all_families()
    G = families[family_name]
    outlet_order = outlet_order_for(G)
    scored = load_scored(family_name)
    screening = act.frozen_screening_demand_set()
    B = jac.zero_sum_basis(len(outlet_order))

    sample = stratified_sample(scored, seed)

    rows = []
    t0 = time.time()
    for stratum, (score8, candidate) in sample:
        mean_err = act.evaluate_candidate(G, candidate, outlet_order, screening, BETA)
        score16 = rescore_16(G, candidate, outlet_order, B)
        rows.append({
            "family": family_name,
            "stratum": stratum,
            "score8": score8,
            "score16": score16,
            "mean_error": mean_err,
            "candidate": str(candidate),
        })
    elapsed = time.time() - t0

    scores8 = [r["score8"] for r in rows]
    scores16 = [r["score16"] for r in rows]
    errs = [r["mean_error"] for r in rows]

    rho8, p8 = spearmanr(scores8, errs)
    rho16, p16 = spearmanr(scores16, errs)

    best_row = min(rows, key=lambda r: r["mean_error"])
    best_candidate = best_row["candidate"]
    best_rank_pos = next(i for i, (s, c) in enumerate(scored) if str(c) == best_candidate)
    best_percentile = 100.0 * best_rank_pos / len(scored)

    print(f"\n=== {family_name} (n_sampled={len(rows)}, elapsed={elapsed:.1f}s) ===")
    print(f"Spearman rho (8-point score vs mean_error): {rho8:.3f} (p={p8:.4f})")
    print(f"Spearman rho (16-point score vs mean_error): {rho16:.3f} (p={p16:.4f})")
    print(f"Best sampled placement's percentile in full 8-point ranking: {best_percentile:.1f} "
          f"(0 = top score, 100 = bottom score)")

    return rows, {"family": family_name, "rho8": rho8, "p8": p8, "rho16": rho16, "p16": p16,
                   "best_percentile": best_percentile, "n_sampled": len(rows)}


if __name__ == "__main__":
    all_rows = []
    summaries = []
    for family, seed in [("murray_tree", 7001), ("grid_lattice", 7002)]:
        rows, summary = audit_family(family, seed)
        all_rows.extend(rows)
        summaries.append(summary)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "actuator_selection")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "proxy_audit_samples.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "stratum", "score8", "score16",
                                           "mean_error", "candidate"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    with open(os.path.join(out_dir, "proxy_audit_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "rho8", "p8", "rho16", "p16",
                                           "best_percentile", "n_sampled"])
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    print("\nSaved results/proxy_audit_samples.csv and results/proxy_audit_summary.csv")
