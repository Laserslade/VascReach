import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import numpy as np
import matplotlib.pyplot as plt

import jacobian as jac
from test_jacobian import random_test_network, pick_controllable_edges


def plot_convergence(outpath):
    rng = np.random.default_rng(22)
    G, outlet_order = random_test_network(rng)
    controllable = pick_controllable_edges(G, rng)
    K = len(controllable)
    z0 = rng.uniform(-0.5, 0.5, size=K)
    y0, dy_dz_analytic = jac.jacobian(G, controllable, outlet_order, z0)

    steps = np.logspace(-2, -9, 15)
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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(steps, errors, marker="o", color="#3b6fa0")
    ax.set_xlabel("finite difference step size h")
    ax.set_ylabel("max abs error vs analytic Jacobian")
    ax.set_title("Analytic vs finite-difference Jacobian, error vs step size")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot_convergence(os.path.join(out_dir, "jacobian_convergence.png"))
    print("Chart saved to", out_dir)
