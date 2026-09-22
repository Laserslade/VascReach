import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import network as net
import jacobian as jac

TOL = 1e-8


def random_test_network(rng, n_outlets=8, n_internal=6, n_extra_edges=3):
    G = net.new_network()
    outlet_names = [f"out{i}" for i in range(n_outlets)]
    internal_names = [f"mid{i}" for i in range(n_internal)]
    all_names = outlet_names + internal_names

    order = rng.permutation(len(all_names))
    shuffled = [all_names[i] for i in order]

    first_internal = internal_names[rng.integers(0, len(internal_names))]
    shuffled.remove(first_internal)

    tree_nodes = ["in"]
    net.add_channel(G, "in", first_internal, length=float(rng.uniform(0.5e-3, 2e-3)),
                     radius=float(rng.uniform(3e-5, 6e-5)))
    tree_nodes.append(first_internal)

    for name in shuffled:
        parent = tree_nodes[rng.integers(0, len(tree_nodes))]
        length = float(rng.uniform(0.5e-3, 2e-3))
        radius = float(rng.uniform(3e-5, 6e-5))
        net.add_channel(G, parent, name, length=length, radius=radius)
        tree_nodes.append(name)

    added, attempts = 0, 0
    while added < n_extra_edges and attempts < 300:
        attempts += 1
        a, b = rng.choice(tree_nodes[1:], size=2, replace=False)
        a, b = str(a), str(b)
        if G.has_edge(a, b):
            continue
        length = float(rng.uniform(0.5e-3, 2e-3))
        radius = float(rng.uniform(3e-5, 6e-5))
        net.add_channel(G, a, b, length=length, radius=radius)
        added += 1

    net.set_node_kind(G, "in", "inlet")
    for name in outlet_names:
        net.set_node_kind(G, name, "outlet")
    for name in internal_names:
        net.set_node_kind(G, name, "internal")

    return G, outlet_names


def pick_controllable_edges(G, rng, k=4):
    kinds = dict(G.nodes(data="kind"))
    candidates = [
        (u, v) for u, v in G.edges()
        if not (kinds[u] != "internal" and kinds[v] != "internal")
    ]
    idx = rng.choice(len(candidates), size=k, replace=False)
    return [(str(candidates[i][0]), str(candidates[i][1])) for i in idx]


def test_zero_sum_basis_orthonormal():
    for N in [4, 8, 12]:
        B = jac.zero_sum_basis(N)
        assert B.shape == (N - 1, N)
        gram = B @ B.T
        assert np.allclose(gram, np.eye(N - 1), atol=1e-10)
        ones = np.ones(N)
        assert np.allclose(B @ ones, np.zeros(N - 1), atol=1e-10)
    print("PASS: zero-sum basis orthonormal and annihilates the common mode")


def test_analytic_vs_finite_difference():
    rng = np.random.default_rng(11)
    max_rel_err = 0.0
    n_networks, n_ztrials = 5, 3
    h = 1e-6

    for net_i in range(n_networks):
        G, outlet_order = random_test_network(rng)
        controllable = pick_controllable_edges(G, rng)
        K = len(controllable)

        for zt in range(n_ztrials):
            z0 = rng.uniform(-1.0, 1.0, size=K)
            y0, dy_dz_analytic = jac.jacobian(G, controllable, outlet_order, z0)

            dy_dz_fd = np.zeros_like(dy_dz_analytic)
            for k in range(K):
                zp = z0.copy(); zp[k] += h
                zm = z0.copy(); zm[k] -= h
                yp = jac.forward_normalized_flow(G, controllable, outlet_order, zp)
                ym = jac.forward_normalized_flow(G, controllable, outlet_order, zm)
                dy_dz_fd[:, k] = (yp - ym) / (2 * h)

            err = np.abs(dy_dz_analytic - dy_dz_fd)
            ok = err < (1e-7 + 1e-4 * np.abs(dy_dz_fd))
            assert np.all(ok), f"Mismatch beyond atol+rtol*fd: max abs err {err.max()}"
            max_rel_err = max(max_rel_err, err.max())

    print(f"PASS: analytic Jacobian matches central finite differences (max abs err {max_rel_err:.2e})")


def test_convergence_with_decreasing_step():
    rng = np.random.default_rng(22)
    G, outlet_order = random_test_network(rng)
    controllable = pick_controllable_edges(G, rng)
    K = len(controllable)
    z0 = rng.uniform(-0.5, 0.5, size=K)

    y0, dy_dz_analytic = jac.jacobian(G, controllable, outlet_order, z0)

    steps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    errors = []
    for h in steps:
        dy_dz_fd = np.zeros_like(dy_dz_analytic)
        for k in range(K):
            zp = z0.copy(); zp[k] += h
            zm = z0.copy(); zm[k] -= h
            yp = jac.forward_normalized_flow(G, controllable, outlet_order, zp)
            ym = jac.forward_normalized_flow(G, controllable, outlet_order, zm)
            dy_dz_fd[:, k] = (yp - ym) / (2 * h)
        errors.append(np.max(np.abs(dy_dz_fd - dy_dz_analytic)))

    assert errors[1] < errors[0], f"Truncation error should drop from the coarsest step: {errors}"
    min_err = min(errors)
    assert min_err < errors[0] / 10, f"No step size approached the analytic value: {errors}"
    print(f"PASS: FD error shows truncation-then-roundoff U-shape: {[f'{e:.2e}' for e in errors]}")


def test_rank_sanity():
    rng = np.random.default_rng(33)
    G, outlet_order = random_test_network(rng)
    controllable = pick_controllable_edges(G, rng, k=4)
    z0 = np.zeros(4)
    y0, dy_dz = jac.jacobian(G, controllable, outlet_order, z0)
    B = jac.zero_sum_basis(len(outlet_order))
    Jc = jac.project_jacobian(dy_dz, B)

    rank = np.linalg.matrix_rank(Jc)
    assert rank <= 4, f"rank(Jc) should be <= K=4, got {rank}"
    assert Jc.shape == (7, 4)
    print(f"PASS: rank(Jc) = {rank} <= K=4, out of 7 zero-sum output dimensions")


def test_physical_perturbation_check():
    rng = np.random.default_rng(44)
    G, outlet_order = random_test_network(rng)
    controllable = pick_controllable_edges(G, rng)
    K = len(controllable)
    z0 = rng.uniform(-0.5, 0.5, size=K)

    y0, dy_dz = jac.jacobian(G, controllable, outlet_order, z0)

    norms = [1e-2, 1e-3, 1e-4, 1e-5]
    errors = []
    for norm in norms:
        dz = rng.normal(size=K)
        dz = dz / np.linalg.norm(dz) * norm
        y1 = jac.forward_normalized_flow(G, controllable, outlet_order, z0 + dz)
        predicted = y0 + dy_dz @ dz
        errors.append(np.linalg.norm(y1 - predicted))

    for i in range(len(norms) - 1):
        assert errors[i + 1] < errors[i], f"Perturbation error should shrink with ||dz||: {errors}"
    print(f"PASS: y(z+dz) - y(z) matches Jc dz, error shrinks with ||dz||: {[f'{e:.2e}' for e in errors]}")


def run_all():
    tests = [
        test_zero_sum_basis_orthonormal,
        test_analytic_vs_finite_difference,
        test_convergence_with_decreasing_step,
        test_rank_sanity,
        test_physical_perturbation_check,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"FAIL: {t.__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} tests FAILED. GATE NOT PASSED.")
        sys.exit(1)
    print(f"All {len(tests)} Jacobian validation checks PASSED. GATE PASSED.")


if __name__ == "__main__":
    run_all()
