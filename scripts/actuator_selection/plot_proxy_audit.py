import sys
import os
import csv

import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["stratum"] = int(r["stratum"])
            r["score8"] = float(r["score8"])
            r["score16"] = float(r["score16"])
            r["mean_error"] = float(r["mean_error"])
            rows.append(r)
    return rows


def plot(outpath, csv_path):
    rows = load_rows(csv_path)
    families = sorted(set(r["family"] for r in rows))

    fig, axes = plt.subplots(1, len(families), figsize=(11, 4.5), sharey=False)
    for ax, fam in zip(axes, families):
        fam_rows = [r for r in rows if r["family"] == fam]
        scores = [r["score8"] for r in fam_rows]
        errs = [r["mean_error"] for r in fam_rows]
        ax.scatter(scores, errs, alpha=0.6, color="#3b6fa0", s=25)
        ax.set_title(fam, fontsize=10)
        ax.set_xlabel("8-point Jacobian score")

    axes[0].set_ylabel("nonlinear mean E* (10-direction screening set)")
    fig.suptitle("Proxy validity audit: stratified full-ranking sample, n=100 each")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "..", "..", "results", "actuator_selection", "proxy_audit_samples.csv")
    out_dir = os.path.join(base, "..", "..", "figures", "actuator_selection")
    os.makedirs(out_dir, exist_ok=True)
    plot(os.path.join(out_dir, "proxy_audit_scatter.png"), csv_path)
    print("Chart saved")
