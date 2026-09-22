import sys
import os
import csv

import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["tercile"] = int(r["tercile"])
            r["delta_p"] = float(r["delta_p"])
            rows.append(r)
    return rows


def plot(outpath, csv_path):
    rows = load_rows(csv_path)
    families = ["symmetric_tree", "murray_tree", "looped_hierarchical", "grid_lattice"]
    terciles = [0, 1, 2]

    fig, axes = plt.subplots(1, len(families), figsize=(14, 4), sharey=True)
    for ax, fam in zip(axes, families):
        data = [[r["delta_p"] for r in rows if r["family"] == fam and r["tercile"] == t]
                for t in terciles]
        ax.boxplot(data, tick_labels=["low", "mid", "high"])
        ax.set_title(fam, fontsize=9)
        ax.set_xlabel("severity tercile")

    axes[0].set_ylabel("Delta_p (max - min E* across 5 batches)")
    fig.suptitle("Optimizer reproducibility across 12 strata (all values near machine epsilon)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "..", "results", "calibration_delta_p.csv")
    out_dir = os.path.join(base, "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot(os.path.join(out_dir, "calibration_delta_p.png"), csv_path)
    print("Chart saved")
