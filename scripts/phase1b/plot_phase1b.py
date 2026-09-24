import sys
import os
import csv

import numpy as np
import matplotlib.pyplot as plt


def load_family_csv(path):
    rows = list(csv.reader(open(path)))
    header = rows[0]
    s_grid = np.array([float(x) for x in rows[-1][2:]])
    data = {}
    for r in rows[1:-1]:
        row_type, fam = r[0], r[1]
        vals = np.array([float(x) if x != "" else np.nan for x in r[2:]])
        data.setdefault(fam, {})[row_type] = vals
    return s_grid, data


def load_network_csv(path):
    rows = list(csv.reader(open(path)))
    s_grid = np.array([float(x) for x in rows[-1][3:]])
    data = {}
    for r in rows[1:-1]:
        row_type, fam, var = r[0], r[1], r[2]
        vals = np.array([float(x) if x != "" else np.nan for x in r[3:]])
        data.setdefault((fam, var), {})[row_type] = vals
    return s_grid, data


def plot(outpath, family_csv, network_csv):
    s_grid, fam_data = load_family_csv(family_csv)
    _, net_data = load_network_csv(network_csv)

    colors = {"symmetric_tree": "#3b6fa0", "murray_tree": "#a03b6f",
              "looped_hierarchical": "#6fa03b", "grid_lattice": "#a08a3b"}

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for (fam, var), d in net_data.items():
        ax.plot(s_grid, d["A"], color=colors[fam], alpha=0.25, linewidth=1)

    for fam, d in fam_data.items():
        ax.plot(s_grid, d["A"], color=colors[fam], linewidth=2.5, label=fam)

    ax.set_xlabel("absolute severity s = ||d - b||_2")
    ax.set_ylabel("A_G(s; epsilon=0.01)")
    ax.set_title("Adaptive accessibility by family (bold) and individual network (faint)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_n_valid(outpath, family_csv):
    s_grid, fam_data = load_family_csv(family_csv)
    colors = {"symmetric_tree": "#3b6fa0", "murray_tree": "#a03b6f",
              "looped_hierarchical": "#6fa03b", "grid_lattice": "#a08a3b"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for fam, d in fam_data.items():
        ax.plot(s_grid, d["n_valid"], color=colors[fam], linewidth=2, label=fam)
    ax.set_xlabel("absolute severity s")
    ax.set_ylabel("n_valid(s)  (out of 60 per family)")
    ax.set_title("Sample size behind each A_G(s) point")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    fam_csv = os.path.join(base, "..", "..", "results", "phase1b", "phase1b_family_accessibility.csv")
    net_csv = os.path.join(base, "..", "..", "results", "phase1b", "phase1b_network_accessibility.csv")
    out_dir = os.path.join(base, "..", "..", "figures", "phase1b")
    os.makedirs(out_dir, exist_ok=True)
    plot(os.path.join(out_dir, "phase1b_accessibility_comparison.png"), fam_csv, net_csv)
    plot_n_valid(os.path.join(out_dir, "phase1b_n_valid.png"), fam_csv)
    print("Charts saved")
