import sys
import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load(path):
    rows = list(csv.DictReader(open(path)))
    A = defaultdict(lambda: defaultdict(list))
    for r in rows:
        A[(r["family"], r["beta_label"])][round(float(r["rho"]), 3)].append(int(r["I_001"]))
    return A


def plot(outpath, csv_path):
    A = load(csv_path)
    families = ["symmetric_tree", "murray_tree", "looped_hierarchical", "grid_lattice"]
    colors = {"symmetric_tree": "#3b6fa0", "murray_tree": "#a03b6f",
              "looped_hierarchical": "#6fa03b", "grid_lattice": "#a08a3b"}
    betas = ["low", "frozen", "high"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, beta_label in zip(axes, betas):
        for fam in families:
            rho_map = A[(fam, beta_label)]
            rhos = sorted(rho_map.keys())
            vals = [np.mean(rho_map[r]) for r in rhos]
            ax.plot(rhos, vals, marker="o", color=colors[fam], label=fam)
        ax.set_title(f"beta = {beta_label}", fontsize=10)
        ax.set_xlabel("rho")

    axes[0].set_ylabel("A_G (fraction accessible)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Accessibility vs actuator authority, 4 canonical networks")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "..", "results", "phase1c_authority_sweep.csv")
    out_dir = os.path.join(base, "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot(os.path.join(out_dir, "phase1c_authority_sweep.png"), csv_path)
    print("Chart saved")
