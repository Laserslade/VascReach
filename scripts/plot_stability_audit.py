import sys
import os
import csv

import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["B"] = int(r["B"])
            r["best_error"] = float(r["best_error"])
            rows.append(r)
    return rows


def plot(outpath, csv_path):
    rows = load_rows(csv_path)
    families = ["symmetric_tree", "murray_tree", "looped_hierarchical", "grid_lattice"]
    budgets = [25, 50, 100]

    fig, axes = plt.subplots(1, len(families), figsize=(14, 4), sharey=False)
    for ax, fam in zip(axes, families):
        data = [[r["best_error"] for r in rows if r["family"] == fam and r["B"] == B] for B in budgets]
        ax.boxplot(data, labels=[str(b) for b in budgets])
        ax.set_title(fam, fontsize=9)
        ax.set_xlabel("budget B")

    axes[0].set_ylabel("best-found nonlinear mean E*")
    fig.suptitle("Placement-search stability: best error across 5 seeds per budget")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "..", "results", "stability_audit.csv")
    out_dir = os.path.join(base, "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot(os.path.join(out_dir, "stability_audit.png"), csv_path)
    print("Chart saved")
