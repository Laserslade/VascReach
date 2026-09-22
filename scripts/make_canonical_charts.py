import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import numpy as np
import matplotlib.pyplot as plt

import canonical as can
import actuator as act
from test_canonical import build_validation_table, outlet_order_for


def plot_baseline_imbalance(outpath):
    rows, _ = build_validation_table()
    names = [r["family"] for r in rows]
    values = [r["E_baseline"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(names, values, color="#3b6fa0")
    ax.set_ylabel("E_baseline")
    ax.set_title("Baseline outlet-flow imbalance by family")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_score_vs_error(outpath):
    families, _ = can.build_all_families()
    screening = act.frozen_screening_demand_set()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"symmetric_tree": "#3b6fa0", "murray_tree": "#a03b6f",
              "looped_hierarchical": "#6fa03b", "grid_lattice": "#a08a3b"}

    for name, G in families.items():
        outlet_order = outlet_order_for(G)
        r = act.select_placements(G, outlet_order, screening)
        scores = [e["score"] for e in r["top_evaluated"]] + [r["poor"]["score"]]
        errs = [e["mean_error"] for e in r["top_evaluated"]] + [r["poor"]["mean_error"]]
        ax.scatter(scores, errs, label=name, color=colors[name])

    ax.set_xlabel("Jacobian placement score (quantile-0.1 sigma_min)")
    ax.set_ylabel("nonlinear mean E* on screening set")
    ax.set_title("Linear screening score versus nonlinear accessibility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot_baseline_imbalance(os.path.join(out_dir, "baseline_imbalance.png"))
    plot_score_vs_error(os.path.join(out_dir, "jacobian_score_vs_nonlinear_error.png"))
    print("Charts saved to", out_dir)
