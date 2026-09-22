import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.optimize import differential_evolution, minimize

import jacobian as jac
import control as ctl
from test_jacobian import random_test_network, pick_controllable_edges

BETA = np.log(2.0)


def test_baseline_recovery():
    rng = np.random.default_rng(100)
    G, outlet_order = random_test_network(rng)
    controllable = pick_controllable_edges(G, rng)
    K = len(controllable)

    y0, _ = jac.jacobian(G, controllable, outlet_order, np.zeros(K))
    result = ctl.solve_inverse(G, controllable, outlet_order, y0, BETA, n_starts=6, seed=1)

    assert result["best_error"] < 1e-8, f"Baseline error too large: {result['best_error']}"
    assert np.linalg.norm(result["best_z"]) < 1e-4, f"z should stay near 0: {result['best_z']}"
    print(f"PASS: baseline recovery, error={result['best_error']:.2e}, |z|={np.linalg.norm(result['best_z']):.2e}")


def test_known_control_recovery():
    rng = np.random.default_rng(101)
    n_trials, max_err = 5, 0.0

    for trial in range(n_trials):
        G, outlet_order = random_test_network(rng)
        controllable = pick_controllable_edges(G, rng)
        K = len(controllable)

        z_true = rng.uniform(-BETA, BETA, size=K)
        d, _ = jac.jacobian(G, controllable, outlet_order, z_true)

        result = ctl.solve_inverse(G, controllable, outlet_order, d, BETA, n_starts=10, seed=2 + trial)
        max_err = max(max_err, result["best_error"])
        assert result["best_error"] < 1e-6, (
            f"Known-control recovery failed on trial {trial}: E*={result['best_error']}"
        )

    print(f"PASS: known-control recovery over {n_trials} trials, max E*={max_err:.2e}")


def test_gradient_validation():
    rng = np.random.default_rng(102)
    G, outlet_order = random_test_network(rng)
    controllable = pick_controllable_edges(G, rng)
    K = len(controllable)
    d = rng.dirichlet(np.ones(8))

    h = 1e-6
    max_err = 0.0
    for _ in range(10):
        z = rng.uniform(-BETA, BETA, size=K)
        L, grad_analytic = ctl.objective_and_grad(z, G, controllable, outlet_order, d)
        grad_fd = np.zeros(K)
        for k in range(K):
            zp = z.copy(); zp[k] += h
            zm = z.copy(); zm[k] -= h
            Lp, _ = ctl.objective_and_grad(zp, G, controllable, outlet_order, d)
            Lm, _ = ctl.objective_and_grad(zm, G, controllable, outlet_order, d)
            grad_fd[k] = (Lp - Lm) / (2 * h)
        err = np.max(np.abs(grad_analytic - grad_fd))
        max_err = max(max_err, err)
        assert err < 1e-6, f"Gradient mismatch: {err}"

    print(f"PASS: analytic gradient of L matches finite differences, max err {max_err:.2e}")


def test_global_search_sanity():
    rng = np.random.default_rng(103)
    n_trials = 3
    for trial in range(n_trials):
        G, outlet_order = random_test_network(rng)
        controllable = pick_controllable_edges(G, rng)
        K = len(controllable)
        z_true = rng.uniform(-BETA, BETA, size=K)
        d, _ = jac.jacobian(G, controllable, outlet_order, z_true)

        local_result = ctl.solve_inverse(G, controllable, outlet_order, d, BETA, n_starts=10, seed=4 + trial)

        def de_objective(z):
            L, _ = ctl.objective_and_grad(z, G, controllable, outlet_order, d)
            return L

        de_res = differential_evolution(
            de_objective, bounds=[(-BETA, BETA)] * K, seed=5 + trial,
            tol=1e-12, maxiter=200, polish=False,
        )
        polished = minimize(
            ctl.objective_and_grad, de_res.x, args=(G, controllable, outlet_order, d),
            jac=True, method="L-BFGS-B", bounds=[(-BETA, BETA)] * K,
        )
        de_error = float(np.sqrt(max(polished.fun, 0.0)))

        assert abs(local_result["best_error"] - de_error) < 1e-5, (
            f"Trial {trial}: multi-start E*={local_result['best_error']}, "
            f"DE+polish E*={de_error}, disagreement too large"
        )
    print(f"PASS: multi-start local search agrees with DE+polish over {n_trials} trials")


def test_normalized_flows_sum_to_one():
    rng = np.random.default_rng(104)
    G, outlet_order = random_test_network(rng)
    controllable = pick_controllable_edges(G, rng)
    K = len(controllable)
    d = rng.dirichlet(np.ones(8))

    result = ctl.solve_inverse(G, controllable, outlet_order, d, BETA, n_starts=4, seed=6)
    total = result["best_normalized_flows"].sum()
    assert abs(total - 1.0) < 1e-10, f"Normalized flows should sum to 1, got {total}"
    print("PASS: normalized outlet flows sum to 1")


def run_all():
    tests = [
        test_baseline_recovery,
        test_known_control_recovery,
        test_gradient_validation,
        test_global_search_sanity,
        test_normalized_flows_sum_to_one,
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
        sys.exit(1)
    print(f"All {len(tests)} control-solver checks PASSED. GATE PASSED.")


if __name__ == "__main__":
    run_all()
