import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import networkx as nx

import network as net
import hydraulics as hyd
import canonical as can


def outlet_order_for(G):
    return sorted([n for n in net.outlet_nodes(G)], key=lambda s: int(s[3:]))


def build_validation_table():
    families, target_volume = can.build_all_families()
    rows = []
    for name, G in families.items():
        outlet_order = outlet_order_for(G)
        result = hyd.solve_network(G, p_inlet=1.0, p_outlet=0.0)
        y = hyd.normalized_outlet_flow_vector(G, result, outlet_order)
        R_eq = hyd.equivalent_resistance(G, result, 1.0, 0.0)
        D = hyd.total_dissipation(result, G)
        V = net.total_channel_volume(G)
        N = len(y)
        E_baseline = float(np.sqrt(np.mean((y - 1.0 / N) ** 2)))

        rows.append({
            "family": name,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "connected": nx.is_connected(G),
            "n_inlets": len(net.inlet_nodes(G)),
            "n_outlets": len(net.outlet_nodes(G)),
            "volume": V,
            "R_eq": R_eq,
            "dissipation": D,
            "outlet_flows": y,
            "E_baseline": E_baseline,
        })
    return rows, target_volume


def test_all_families_connected_and_valid_counts():
    rows, _ = build_validation_table()
    for r in rows:
        assert r["connected"], f"{r['family']} is not connected"
        assert r["n_inlets"] == 1, f"{r['family']} has {r['n_inlets']} inlets"
        assert r["n_outlets"] == 8, f"{r['family']} has {r['n_outlets']} outlets"
    print("PASS: all four families connected, 1 inlet, 8 outlets")


def test_volumes_matched():
    rows, target_volume = build_validation_table()
    for r in rows:
        rel_err = abs(r["volume"] - target_volume) / target_volume
        assert rel_err < 1e-9, f"{r['family']} volume mismatch: {rel_err}"
    print(f"PASS: all four families matched to target volume {target_volume:.4e} m^3")


def test_outlet_flows_nonnegative_and_sum_to_one():
    rows, _ = build_validation_table()
    for r in rows:
        y = r["outlet_flows"]
        assert np.all(y >= -1e-12), f"{r['family']} has negative outlet flow"
        assert abs(y.sum() - 1.0) < 1e-9, f"{r['family']} outlet flows do not sum to 1"
    print("PASS: all outlet flow vectors nonnegative and normalized")


def print_table_and_flag_imbalance():
    rows, _ = build_validation_table()
    print()
    print(f"{'family':<22}{'N_V':>5}{'N_E':>5}{'R_eq':>14}{'D':>14}{'E_baseline':>13}")
    for r in rows:
        print(f"{r['family']:<22}{r['n_nodes']:>5}{r['n_edges']:>5}"
              f"{r['R_eq']:>14.4e}{r['dissipation']:>14.4e}{r['E_baseline']:>13.4e}")
    print()

    max_E = max(r["E_baseline"] for r in rows)
    if max_E > 0.05:
        worst = max(rows, key=lambda r: r["E_baseline"])
        print(f"FLAG: baseline imbalance is nontrivial. Worst family: {worst['family']} "
              f"(E_baseline={worst['E_baseline']:.4e})")
    else:
        print(f"No family shows severe baseline imbalance (max E_baseline={max_E:.4e})")


def run_all():
    tests = [
        test_all_families_connected_and_valid_counts,
        test_volumes_matched,
        test_outlet_flows_nonnegative_and_sum_to_one,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"FAIL: {t.__name__}: {e}")
    print_table_and_flag_imbalance()
    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} tests FAILED.")
        sys.exit(1)
    print(f"All {len(tests)} canonical-network checks PASSED.")


if __name__ == "__main__":
    run_all()
