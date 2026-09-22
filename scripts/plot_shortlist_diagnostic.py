import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["n_top"] = int(r["n_top"])
            r["best_error"] = float(r["best_error"])
            r["poor_error"] = float(r["poor_error"])
            rows.append(r)
    return rows


def plot(outpath, csv_path):
    rows = load_rows(csv_path)
    families = sorted(set(r["family"] for r in rows))

    fig, axes = plt.subplots(1, len(families), figsize=(14, 4), sharey=True)
    for ax, fam in zip(axes, families):
        fam_rows = [r for r in rows if r["family"] == fam]
        fam_rows.sort(key=lambda r: r["n_top"])
        n_tops = [r["n_top"] for r in fam_rows]
        best = [r["best_error"] for r in fam_rows]
        poor = fam_rows[0]["poor_error"]

        ax.plot(n_tops, best, marker="o", color="#3b6fa0", label="best of shortlist")
        ax.axhline(poor, color="#a03b3b", linestyle="--", label="poor arm")
        ax.set_title(fam, fontsize=9)
        ax.set_xlabel("shortlist size")
        ax.set_xticks(n_tops)

    axes[0].set_ylabel("nonlinear mean E*")
    axes[0].legend(fontsize=8)
    fig.suptitle("Shortlist-breadth diagnostic: does a wider shortlist beat the poor arm")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "shortlist_diagnostic.csv")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot(os.path.join(out_dir, "shortlist_diagnostic.png"), csv_path)
    print("Chart saved")
