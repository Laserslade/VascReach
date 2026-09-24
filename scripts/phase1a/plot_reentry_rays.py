import sys
import os
import csv

import matplotlib.pyplot as plt


def load_ray(path, direction_id):
    pts = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["family"] == "symmetric_tree" and r["variant"] == "1" and int(r["direction_id"]) == direction_id:
                pts.append((float(r["rho"]), float(r["best_error"])))
    pts.sort()
    return pts


def plot(outpath, csv_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, did in zip(axes, [1, 3]):
        pts = load_ray(csv_path, did)
        rhos = [p[0] for p in pts]
        errs = [p[1] for p in pts]
        colors = ["#3b6fa0" if e <= 0.01 else "#a03b3b" for e in errs]
        ax.scatter(rhos, errs, c=colors, s=20)
        ax.axhline(0.01, color="black", linestyle="--", linewidth=1, label="epsilon = 0.01")
        ax.set_title(f"symmetric_tree variant 1, direction {did}", fontsize=9)
        ax.set_xlabel("rho = s / s_max")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("E*(s)  (blue = accessible, red = not)")
    fig.suptitle("Verified non-interval-like accessibility: two rays with confirmed 0->1 re-entry")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "..", "..", "results", "phase1a", "phase1a_dense_grid.csv")
    out_dir = os.path.join(base, "..", "..", "figures", "phase1a")
    os.makedirs(out_dir, exist_ok=True)
    plot(os.path.join(out_dir, "phase1a_reentry_rays.png"), csv_path)
    print("Chart saved")
