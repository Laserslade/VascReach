import sys
import os
import csv

import numpy as np
import matplotlib.pyplot as plt


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["K"] = int(r["K"])
        r["E0_star"] = float(r["E0_star"])
        r["regime"] = r["net_id"].rsplit("_", 1)[0]
    return rows


def plot(outpath, csv_path):
    rows = load(csv_path)
    regimes = ["tree", "loopy_hierarchical", "lattice", "random_spatial"]
    colors = {"tree": "#3b6fa0", "loopy_hierarchical": "#6fa03b",
              "lattice": "#a08a3b", "random_spatial": "#a03b6f"}
    conditions = [(4, "log2"), (4, "log4"), (6, "log2"), (6, "log4"), (8, "log2"), (8, "log4")]

    fig, ax = plt.subplots(figsize=(9, 5))
    for regime in regimes:
        means = []
        for K, beta_label in conditions:
            vals = [r["E0_star"] for r in rows if r["regime"] == regime
                    and r["K"] == K and r["beta_label"] == beta_label]
            means.append(np.mean(vals))
        x = range(len(conditions))
        ax.plot(x, means, marker="o", color=colors[regime], label=regime)

    ax.axhline(0.01, color="black", linestyle="--", linewidth=1, label="epsilon=0.01")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([f"K={k}\n{b}" for k, b in conditions])
    ax.set_ylabel("mean E0* = min_z E(f(G,z), b)")
    ax.set_title("Baseline correctability by regime across actuator resource grid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "..", "..", "results", "phase2", "phase2_resource_calibration.csv")
    out_dir = os.path.join(base, "..", "..", "figures", "phase2")
    os.makedirs(out_dir, exist_ok=True)
    plot(os.path.join(out_dir, "phase2_resource_calibration.png"), csv_path)
    print("Chart saved")
