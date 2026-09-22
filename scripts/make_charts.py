import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import numpy as np
import matplotlib.pyplot as plt

import demand as dem
from test_loop_stress import run as run_loop_stress


def plot_loop_stress_flows(outpath):
    G, result = run_loop_stress()
    edges = list(result["flows"].keys())
    flows = [result["flows"][e] for e in edges]
    labels = [f"{u}-{v}" for u, v in edges]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, flows, color="#3b6fa0")
    ax.set_ylabel("Q (m^3/s)")
    ax.set_title("Loop stress test: edge flows")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_demand_families(outpath):
    positions = dem.outlet_positions_uniform(8)
    delta_x = dem.median_outlet_spacing(positions)
    b = np.full(8, 1.0 / 8)
    rng = np.random.default_rng(123)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fam in ["localized", "regional", "multifocal"]:
        d = dem.generate_demand_direction(rng, positions, fam, delta_x, 0, 123)
        dvec = dem.demand_at_severity(b, d.v_hat, d.s_max * 0.8)
        ax.plot(range(1, 9), dvec, marker="o", label=fam)

    ax.set_xlabel("Outlet index")
    ax.set_ylabel("d_i")
    ax.set_title("Demand vectors at s = 0.8 * s_max, by pattern family")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot_loop_stress_flows(os.path.join(out_dir, "loop_stress_flows.png"))
    plot_demand_families(os.path.join(out_dir, "demand_pattern_families.png"))
    print("Charts saved to", out_dir)
