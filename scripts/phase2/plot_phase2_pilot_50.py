import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import topology_gen as tg
import canonical as can
from phase2_pilot_50 import MECHANISMS

sys.path.insert(0, os.path.dirname(__file__))


def plot_distributions(outpath, csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    colors = {"tree": "#3b6fa0", "loopy_hierarchical": "#6fa03b",
              "lattice": "#a08a3b", "random_spatial": "#a03b6f"}

    fields = ["cycle_rank", "R_eq", "path_redundancy", "E_baseline"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, field in zip(axes, fields):
        data = [[float(r[field]) for r in rows if r["mechanism"] == m] for m in MECHANISMS]
        bp = ax.boxplot(data, tick_labels=MECHANISMS)
        ax.set_title(field, fontsize=10)
        ax.tick_params(axis="x", rotation=25)
        if field == "R_eq":
            ax.set_yscale("log")
    fig.suptitle("Phase 2 pilot: structural/hydraulic descriptor spread by mechanism (n=13 each)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_example_layouts(outpath, target_volume):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, mech in zip(axes, MECHANISMS):
        G, meta = tg.generate_network(mech, 5, target_volume)
        pos = {}
        outlets = tg.outlet_points()
        pos["in"] = (0.0, 0.0)
        for i in range(8):
            pos[f"out{i}"] = tuple(outlets[i])

        for n in G.nodes():
            if n not in pos:
                pos[n] = None

        missing = [n for n in G.nodes() if pos[n] is None]
        if missing:
            spring_pos = nx.spring_layout(G, pos={k: v for k, v in pos.items() if v is not None},
                                           fixed=[k for k, v in pos.items() if v is not None], seed=1)
            for n in missing:
                pos[n] = spring_pos[n]

        node_colors = []
        for n in G.nodes():
            if n == "in":
                node_colors.append("red")
            elif n.startswith("out"):
                node_colors.append("blue")
            else:
                node_colors.append("gray")

        nx.draw(G, pos=pos, ax=ax, node_size=40, node_color=node_colors, edge_color="black",
                width=0.8, with_labels=False)
        ax.set_title(f"{mech}\ncycle_rank={G.number_of_edges()-G.number_of_nodes()+1}", fontsize=9)

    fig.suptitle("Example network from each mechanism (red=inlet, blue=outlets, gray=internal)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    out_dir = os.path.join(base, "..", "..", "figures", "phase2")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(base, "..", "..", "results", "phase2", "phase2_pilot_50_descriptors.csv")
    plot_distributions(os.path.join(out_dir, "phase2_pilot_50_distributions.png"), csv_path)

    _, tv = can.build_all_families()
    plot_example_layouts(os.path.join(out_dir, "phase2_pilot_50_examples.png"), tv)

    print("Charts saved")
