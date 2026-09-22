import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import numpy as np
import networkx as nx

import network as net
import hydraulics as hyd
import canonical as can
import topology_gen as tg

N_PER_MECHANISM = 13
MECHANISMS = ["tree", "loopy_hierarchical", "lattice", "random_spatial"]


def outlet_order_for(G):
    return sorted([n for n in net.outlet_nodes(G)], key=lambda s: int(s[3:]))


def build_pilot_50():
    _, target_volume = can.build_all_families()
    networks = []
    for mech in MECHANISMS:
        count = 0
        seed = 0
        attempts = 0
        while count < N_PER_MECHANISM and attempts < N_PER_MECHANISM * 5:
            attempts += 1
            seed += 1
            G, meta = tg.generate_network(mech, seed + 100000 * MECHANISMS.index(mech), target_volume)
            if G is None:
                continue
            networks.append((G, meta))
            count += 1
    return networks, target_volume


def compute_descriptors(networks, target_volume):
    rows = []
    hashes = {}
    for G, meta in networks:
        outlet_order = outlet_order_for(G)
        N_V = G.number_of_nodes()
        N_E = G.number_of_edges()
        cycle_rank = N_E - N_V + 1
        connected = nx.is_connected(G)
        V = net.total_channel_volume(G)

        result = hyd.solve_network(G, p_inlet=1.0, p_outlet=0.0)
        R_eq = hyd.equivalent_resistance(G, result, 1.0, 0.0)
        y = hyd.normalized_outlet_flow_vector(G, result, outlet_order)
        E_baseline = float(np.sqrt(np.mean((y - 1.0 / 8) ** 2)))
        path_redundancy = N_E / (N_V - 1)
        mean_degree = 2.0 * N_E / N_V

        wl_hash = nx.weisfeiler_lehman_graph_hash(G)
        hashes.setdefault(wl_hash, []).append(meta["mechanism"])

        rows.append({
            "mechanism": meta["mechanism"], "seed": meta["seed"],
            "n_internal": meta["n_internal"], "retain_fraction": meta["retain_fraction"],
            "N_V": N_V, "N_E": N_E, "cycle_rank": cycle_rank, "connected": connected,
            "volume": V, "vol_rel_err": abs(V - target_volume) / target_volume,
            "R_eq": R_eq, "E_baseline": E_baseline,
            "path_redundancy": path_redundancy, "mean_degree": mean_degree,
            "wl_hash": wl_hash,
        })

    duplicates = {h: v for h, v in hashes.items() if len(v) > 1}
    return rows, duplicates


if __name__ == "__main__":
    networks, target_volume = build_pilot_50()
    print(f"Generated {len(networks)} networks ({N_PER_MECHANISM} per mechanism x {len(MECHANISMS)})")

    rows, duplicates = compute_descriptors(networks, target_volume)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "phase2_pilot_50_descriptors.csv"), "w", newline="") as f:
        fields = ["mechanism", "seed", "n_internal", "retain_fraction", "N_V", "N_E",
                   "cycle_rank", "connected", "volume", "vol_rel_err", "R_eq", "E_baseline",
                   "path_redundancy", "mean_degree", "wl_hash"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_connected = sum(1 for r in rows if r["connected"])
    max_vol_err = max(r["vol_rel_err"] for r in rows)
    print(f"Connected: {n_connected}/{len(rows)}")
    print(f"Max volume relative error: {max_vol_err:.2e}")
    print(f"Duplicate (WL-hash) graphs found: {len(duplicates)}")
    for h, mechs in duplicates.items():
        print(f"  hash {h[:12]}...: {mechs}")

    print()
    for mech in MECHANISMS:
        vals = [r for r in rows if r["mechanism"] == mech]
        cr = [r["cycle_rank"] for r in vals]
        req = [r["R_eq"] for r in vals]
        pr = [r["path_redundancy"] for r in vals]
        eb = [r["E_baseline"] for r in vals]
        print(f"{mech:<20} cycle_rank[{min(cr)},{max(cr)}] "
              f"R_eq[{min(req):.2e},{max(req):.2e}] "
              f"path_redundancy[{min(pr):.2f},{max(pr):.2f}] "
              f"E_baseline[{min(eb):.2e},{max(eb):.2e}]")
