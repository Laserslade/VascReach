import sys
import os
import csv

import numpy as np
import matplotlib.pyplot as plt


def plot_morphology(outpath, csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"localized": "#3b6fa0", "regional": "#a03b6f", "multifocal": "#6fa03b"}
    for fam in ["localized", "regional", "multifocal"]:
        pts = [(float(r["rho"]), float(r["A"])) for r in rows if r["pattern_family"] == fam]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=colors[fam], label=fam)
    ax.set_xlabel("rho")
    ax.set_ylabel("A(s) pooled across all 12 networks")
    ax.set_title("Accessibility by demand morphology")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_multifocal(outpath, csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    sep = [float(r["separation_over_dx"]) for r in rows]
    width = [float(r["width_over_dx"]) for r in rows]
    err = [float(r["E_at_rho0.2"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(sep, err, color="#3b6fa0", alpha=0.7)
    axes[0].axhline(0.01, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("center separation / delta_x")
    axes[0].set_ylabel("E* at rho=0.2")

    axes[1].scatter(width, err, color="#a03b6f", alpha=0.7)
    axes[1].axhline(0.01, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("width sigma / delta_x")
    axes[1].set_ylabel("E* at rho=0.2")

    fig.suptitle("Multifocal geometry vs tracking error (n=72 rays, rho=0.2)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    out_dir = os.path.join(base, "..", "..", "figures", "phase1c")
    os.makedirs(out_dir, exist_ok=True)
    plot_morphology(os.path.join(out_dir, "phase1c_morphology.png"),
                     os.path.join(base, "..", "..", "results", "phase1c", "phase1c_morphology.csv"))
    plot_multifocal(os.path.join(out_dir, "phase1c_multifocal_geometry.png"),
                     os.path.join(base, "..", "..", "results", "phase1c", "phase1c_multifocal_geometry.csv"))
    print("Charts saved")
