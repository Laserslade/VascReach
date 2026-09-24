import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import networkx as nx

import network as net
import hydraulics as hyd
import canonical as can
import topology_gen as tg

N_PER_REGIME = 25
REGIMES = ["tree", "loopy_hierarchical", "lattice", "random_spatial"]


def outlet_order_for(G):
    return sorted([n for n in net.outlet_nodes(G)], key=lambda s: int(s[3:]))


def build_ensemble_100():
    _, target_volume = can.build_all_families()
    networks = []
    for regime in REGIMES:
        count, seed, attempts = 0, 0, 0
        while count < N_PER_REGIME and attempts < N_PER_REGIME * 5:
            attempts += 1
            seed += 1
            full_seed = seed + 100000 * REGIMES.index(regime)
            G, meta = tg.generate_network(regime, full_seed, target_volume)
            if G is None:
                continue
            net_id = f"{regime}_{count:03d}"
            networks.append((net_id, G, meta))
            count += 1
    return networks, target_volume


def compute_descriptors(networks, target_volume):
    rows = []
    for net_id, G, meta in networks:
        outlet_order = outlet_order_for(G)
        N_V, N_E = G.number_of_nodes(), G.number_of_edges()
        cycle_rank = N_E - N_V + 1
        result = hyd.solve_network(G, p_inlet=1.0, p_outlet=0.0)
        R_eq = hyd.equivalent_resistance(G, result, 1.0, 0.0)
        y = hyd.normalized_outlet_flow_vector(G, result, outlet_order)
        E_baseline = float(np.sqrt(np.mean((y - 1.0 / 8) ** 2)))
        rows.append({
            "net_id": net_id, "regime": meta["mechanism"], "seed": meta["seed"],
            "n_internal": meta["n_internal"], "retain_fraction": meta["retain_fraction"],
            "N_V": N_V, "N_E": N_E, "cycle_rank": cycle_rank,
            "connected": nx.is_connected(G), "R_eq": R_eq, "E_baseline": E_baseline,
        })
    return rows


if __name__ == "__main__":
    networks, target_volume = build_ensemble_100()
    rows = compute_descriptors(networks, target_volume)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "phase2_ensemble_100.csv"), "w", newline="") as f:
        fields = ["net_id", "regime", "seed", "n_internal", "retain_fraction", "N_V", "N_E",
                   "cycle_rank", "connected", "R_eq", "E_baseline"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Generated {len(rows)} networks, all connected: {all(r['connected'] for r in rows)}")
    for regime in REGIMES:
        cr = [r["cycle_rank"] for r in rows if r["regime"] == regime]
        print(f"{regime}: cycle_rank range [{min(cr)}, {max(cr)}]")
