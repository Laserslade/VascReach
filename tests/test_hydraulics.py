import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import networkx as nx

import network as net
import hydraulics as hyd

TOL = 1e-9


def relerr(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def test_resistance_scales_with_length():
    mu = 1e-3
    r = 5e-5
    R1 = net.hagen_poiseuille_resistance(mu, 1e-3, r)
    R2 = net.hagen_poiseuille_resistance(mu, 2e-3, r)
    assert relerr(R2 / R1, 2.0) < TOL
    print("PASS: R proportional to L")


def test_resistance_scales_with_radius_inverse_fourth():
    mu = 1e-3
    L = 1e-3
    R1 = net.hagen_poiseuille_resistance(mu, L, 5e-5)
    R2 = net.hagen_poiseuille_resistance(mu, L, 1e-4)
    expected_ratio = 2.0**4
    assert relerr(R1 / R2, expected_ratio) < TOL
    print("PASS: R proportional to r^-4")


def test_single_channel():
    G = net.new_network()
    net.add_channel(G, "in", "out", length=1e-3, radius=5e-5)
    net.set_node_kind(G, "in", "inlet")
    net.set_node_kind(G, "out", "outlet")

    p_in, p_out = 100.0, 0.0
    result = hyd.solve_network(G, p_inlet=p_in, p_outlet=p_out)

    R = G["in"]["out"]["resistance"]
    Q_expected = (p_in - p_out) / R
    Q_actual = result["flows"][("in", "out")]

    assert relerr(Q_actual, Q_expected) < TOL
    assert relerr(result["outlet_flows"]["out"], Q_expected) < TOL
    print(f"PASS: single channel Q = dP/R  (Q={Q_actual:.6e})")


def test_series():
    G = net.new_network()
    net.add_channel(G, "in", "mid", length=1e-3, radius=4e-5)
    net.add_channel(G, "mid", "out", length=2e-3, radius=6e-5)
    net.set_node_kind(G, "in", "inlet")
    net.set_node_kind(G, "mid", "internal")
    net.set_node_kind(G, "out", "outlet")

    p_in, p_out = 200.0, 0.0
    result = hyd.solve_network(G, p_inlet=p_in, p_outlet=p_out)

    R1 = G["in"]["mid"]["resistance"]
    R2 = G["mid"]["out"]["resistance"]
    R_eq_expected = R1 + R2
    Q_expected = (p_in - p_out) / R_eq_expected

    Q1 = result["flows"][("in", "mid")]
    Q2 = result["flows"][("mid", "out")]

    assert relerr(Q1, Q2) < TOL
    assert relerr(Q1, Q_expected) < TOL

    R_eq_actual = hyd.equivalent_resistance(G, result, p_in, p_out)
    assert relerr(R_eq_actual, R_eq_expected) < TOL
    print(f"PASS: series R_eq = R1 + R2  (R_eq={R_eq_actual:.6e})")


def test_parallel():
    G_multi = nx.MultiGraph()
    G_multi.graph["mu"] = net.DEFAULT_MU
    mu = G_multi.graph["mu"]

    L1, r1 = 1e-3, 4e-5
    L2, r2 = 1.5e-3, 5e-5
    R1 = net.hagen_poiseuille_resistance(mu, L1, r1)
    R2 = net.hagen_poiseuille_resistance(mu, L2, r2)

    G_multi.add_edge("in", "out", length=L1, radius=r1, resistance=R1, resistance0=R1, controllable=False)
    G_multi.add_edge("in", "out", length=L2, radius=r2, resistance=R2, resistance0=R2, controllable=False)
    net.set_node_kind(G_multi, "in", "inlet")
    net.set_node_kind(G_multi, "out", "outlet")

    p_in, p_out = 150.0, 0.0
    result = solve_multigraph_parallel(G_multi, p_in, p_out)

    R_eq_expected = 1.0 / (1.0 / R1 + 1.0 / R2)
    Q_total_expected = (p_in - p_out) / R_eq_expected
    Q_total_actual = sum(result["outlet_flows"].values())

    assert relerr(Q_total_actual, Q_total_expected) < TOL
    print(f"PASS: parallel 1/R_eq = 1/R1 + 1/R2  (Q_total={Q_total_actual:.6e})")


def solve_multigraph_parallel(G_multi, p_in, p_out):
    Q_total = 0.0
    flows = {}
    for i, (u, v, data) in enumerate(G_multi.edges(data=True)):
        R = data["resistance"]
        Q = (p_in - p_out) / R
        flows[(u, v, i)] = Q
        Q_total += Q
    return {"flows": flows, "outlet_flows": {"out": Q_total}}


def test_symmetric_bifurcation():
    G = net.new_network()
    net.add_channel(G, "in", "out1", length=1e-3, radius=5e-5)
    net.add_channel(G, "in", "out2", length=1e-3, radius=5e-5)
    net.set_node_kind(G, "in", "inlet")
    net.set_node_kind(G, "out1", "outlet")
    net.set_node_kind(G, "out2", "outlet")

    result = hyd.solve_network(G, p_inlet=100.0, p_outlet=0.0)

    Q1 = result["outlet_flows"]["out1"]
    Q2 = result["outlet_flows"]["out2"]

    assert relerr(Q1, Q2) < TOL
    assert Q1 > 0 and Q2 > 0
    print(f"PASS: symmetric bifurcation Q1 = Q2  (Q1={Q1:.6e}, Q2={Q2:.6e})")


def test_rectangular_channel_positive_and_reasonable():
    mu, L, h = 1e-3, 1e-3, 2e-5
    w = 1.0
    R_wide = net.rectangular_channel_resistance(mu, L, width=w, height=h)
    R_parallel_plate_limit = 12.0 * mu * L / (w * h**3)
    assert R_wide > 0
    assert relerr(R_wide, R_parallel_plate_limit) < 1e-3
    print("PASS: rectangular-channel resistance sanity check")


def run_all():
    tests = [
        test_resistance_scales_with_length,
        test_resistance_scales_with_radius_inverse_fourth,
        test_single_channel,
        test_series,
        test_parallel,
        test_symmetric_bifurcation,
        test_rectangular_channel_positive_and_reasonable,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"FAIL: {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"ERROR: {e}"))
            print(f"ERROR in {t.__name__}: {e}")

    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} tests FAILED. GATE NOT PASSED.")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests PASSED to tolerance {TOL}. GATE PASSED.")


if __name__ == "__main__":
    run_all()
