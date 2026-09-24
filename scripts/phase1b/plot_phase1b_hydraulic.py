import sys
import os
import csv

import numpy as np
import matplotlib.pyplot as plt


def load(path):
    rows = list(csv.DictReader(open(path)))
    by_fam = {}
    for r in rows:
        by_fam.setdefault(r["family"], []).append(float(r["R_eq"]))
    return by_fam


def plot(outpath, csv_path):
    by_fam = load(csv_path)
    families = ["symmetric_tree", "murray_tree", "looped_hierarchical", "grid_lattice"]
    colors = {"symmetric_tree": "#3b6fa0", "murray_tree": "#a03b6f",
              "looped_hierarchical": "#6fa03b", "grid_lattice": "#a08a3b"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, fam in enumerate(families):
        vals = by_fam[fam]
        ax.scatter([i] * len(vals), vals, color=colors[fam], s=60, zorder=3)
        ax.hlines(np.median(vals), i - 0.2, i + 0.2, color="black", zorder=4)

    ax.set_yscale("log")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=15)
    ax.set_ylabel("R_eq (Pa . s / m^3), log scale")
    ax.set_title("Conventional hydraulic quality: 3 variants per family, median marked")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "..", "..", "results", "phase1b", "phase1b_hydraulic_quality.csv")
    out_dir = os.path.join(base, "..", "..", "figures", "phase1b")
    os.makedirs(out_dir, exist_ok=True)
    plot(os.path.join(out_dir, "phase1b_hydraulic_quality.png"), csv_path)
    print("Chart saved")
