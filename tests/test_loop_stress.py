import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import network as net
import hydraulics as hyd

TOL_ABS = 1e-10


def build_looped_network():
    G = net.new_network()
    net.add_channel(G, "in", "s", length=1e-3, radius=5e-5)
    net.add_channel(G, "s", "j", length=1.2e-3, radius=4e-5)
    net.add_channel(G, "s", "m", length=1.5e-3, radius=3.5e-5)
    net.add_channel(G, "m", "j", length=1.5e-3, radius=3.5e-5)
    net.add_channel(G, "j", "out1", length=1e-3, radius=4.5e-5)
    net.add_channel(G, "j", "out2", length=1.3e-3, radius=4.2e-5)

    for node, kind in [
        ("in", "inlet"),
        ("s", "internal"),
        ("m", "internal"),
        ("j", "internal"),
        ("out1", "outlet"),
        ("out2", "outlet"),
    ]:
        net.set_node_kind(G, node, kind)
    return G


def run():
    G = build_looped_network()
    result = hyd.solve_network(G, p_inlet=100.0, p_outlet=0.0)
    pressures = result["pressures"]
    flows = result["flows"]

    free_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "internal"]

    max_imbalance = 0.0
    for v in free_nodes:
        net_flow = 0.0
        for u in G.neighbors(v):
            if (u, v) in flows:
                net_flow += flows[(u, v)]
            else:
                net_flow -= flows[(v, u)]
        max_imbalance = max(max_imbalance, abs(net_flow))
    assert max_imbalance < TOL_ABS
    print(f"PASS: flow conservation at all free nodes (max imbalance {max_imbalance:.3e})")

    max_antisym_err = 0.0
    for (u, v), Q_uv in flows.items():
        R = G[u][v]["resistance"]
        Q_vu_direct = (pressures[v] - pressures[u]) / R
        max_antisym_err = max(max_antisym_err, abs(Q_vu_direct - (-Q_uv)))
    assert max_antisym_err < TOL_ABS
    print(f"PASS: Q_ij = -Q_ji (max error {max_antisym_err:.3e})")

    Q_in = flows[("in", "s")]
    Q_out_total = sum(result["outlet_flows"].values())
    global_err = abs(Q_in - Q_out_total)
    assert global_err < TOL_ABS
    print(f"PASS: Q_in - sum(Q_out) = {global_err:.3e}")

    min_product = min(Q * (pressures[u] - pressures[v]) for (u, v), Q in flows.items())
    assert min_product >= -TOL_ABS
    print(f"PASS: all edges satisfy Q_ij (P_i - P_j) >= 0 (min {min_product:.3e})")

    print()
    print("Loop stress test PASSED.")
    return G, result


if __name__ == "__main__":
    run()
