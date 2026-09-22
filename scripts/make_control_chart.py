import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import numpy as np
import matplotlib.pyplot as plt

import jacobian as jac
import control as ctl
from test_jacobian import random_test_network, pick_controllable_edges

BETA = np.log(2.0)


def plot_recovery_errors(outpath, n_trials=25):
    rng = np.random.default_rng(200)
    errors = []
    for trial in range(n_trials):
        G, outlet_order = random_test_network(rng)
        controllable = pick_controllable_edges(G, rng)
        K = len(controllable)
        z_true = rng.uniform(-BETA, BETA, size=K)
        d, _ = jac.jacobian(G, controllable, outlet_order, z_true)
        result = ctl.solve_inverse(G, controllable, outlet_order, d, BETA, n_starts=10, seed=300 + trial)
        errors.append(result["best_error"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(range(1, n_trials + 1), errors, color="#3b6fa0")
    ax.set_yscale("log")
    ax.set_xlabel("trial")
    ax.set_ylabel("E* (log scale)")
    ax.set_title("Known-control recovery: E* across random trials")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot_recovery_errors(os.path.join(out_dir, "control_recovery_errors.png"))
    print("Chart saved to", out_dir)
