from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

import network as net

DOMAIN_RADIUS = 1.0
N_OUTLETS = 8


def outlet_points(n_outlets: int = N_OUTLETS, radius: float = DOMAIN_RADIUS):
    angles = np.linspace(0, 2 * np.pi, n_outlets, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def _radial_branching_points(rng, n_points, n_levels=3, radius=DOMAIN_RADIUS):
    levels, angles = [], []

    def recurse(lo, hi, depth, remaining):
        if remaining <= 0 or depth > n_levels:
            return
        angles.append(rng.uniform(lo, hi))
        levels.append(depth)
        remaining -= 1
        if remaining <= 0:
            return
        mid = (lo + hi) / 2
        left_n = remaining // 2
        recurse(lo, mid, depth + 1, left_n)
        recurse(mid, hi, depth + 1, remaining - left_n)

    recurse(0, 2 * np.pi, 1, n_points)
    levels = np.array(levels)
    angles = np.array(angles)
    radii = (levels / n_levels) * radius * 0.8 + rng.uniform(0.02, 0.08, size=len(levels))
    pts = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    return pts


def _jittered_grid_points(rng, n_points, radius=DOMAIN_RADIUS):
    side = max(2, int(round(np.sqrt(n_points))))
    lin = np.linspace(-radius * 0.75, radius * 0.75, side)
    xx, yy = np.meshgrid(lin, lin)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    pts = pts[np.linalg.norm(pts, axis=1) < radius * 0.85]
    jitter = rng.uniform(-radius * 0.08, radius * 0.08, size=pts.shape)
    return pts + jitter


def _random_points(rng, n_points, radius=DOMAIN_RADIUS):
    r = radius * 0.85 * np.sqrt(rng.uniform(0, 1, size=n_points))
    theta = rng.uniform(0, 2 * np.pi, size=n_points)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


MECHANISM_CONFIG = {
    "tree": {"placement": "radial_branching", "retain_range": (0.0, 0.0), "n_internal_range": (6, 14)},
    "loopy_hierarchical": {"placement": "radial_branching", "retain_range": (0.1, 0.4), "n_internal_range": (6, 14)},
    "lattice": {"placement": "jittered_grid", "retain_range": (0.3, 0.6), "n_internal_range": (9, 20)},
    "random_spatial": {"placement": "random", "retain_range": (0.0, 0.5), "n_internal_range": (6, 16)},
}


def _place_internal_points(mode, rng, n_points):
    if mode == "radial_branching":
        return _radial_branching_points(rng, n_points)
    if mode == "jittered_grid":
        return _jittered_grid_points(rng, n_points)
    if mode == "random":
        return _random_points(rng, n_points)
    raise ValueError(f"Unknown placement mode: {mode}")


def _build_graph_from_points(all_points, node_names, inlet_idx, outlet_idxs, retain_fraction, rng,
                              target_volume, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            tri = Delaunay(all_points)
        except Exception:
            continue

        edge_set = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                edge_set.add((min(a, b), max(a, b)))

        if len(edge_set) < len(all_points) - 1:
            continue

        G_full = nx.Graph()
        for i in range(len(all_points)):
            G_full.add_node(i)
        for a, b in edge_set:
            dist = float(np.linalg.norm(all_points[a] - all_points[b]))
            G_full.add_edge(a, b, weight=dist)

        if not nx.is_connected(G_full):
            continue

        mst = nx.minimum_spanning_tree(G_full, weight="weight")
        non_mst_edges = [e for e in G_full.edges() if not mst.has_edge(*e)]
        rng.shuffle(non_mst_edges)
        n_extra = int(round(retain_fraction * len(non_mst_edges)))
        extra_edges = non_mst_edges[:n_extra]

        G = net.new_network()
        for a, b in list(mst.edges()) + extra_edges:
            u, v = node_names[a], node_names[b]
            length = max(float(np.linalg.norm(all_points[a] - all_points[b])), 1e-4) * 1e-3
            net.add_channel(G, u, v, length=length, radius=5e-5)

        for i, name in enumerate(node_names):
            if i == inlet_idx:
                net.set_node_kind(G, name, "inlet")
            elif i in outlet_idxs:
                net.set_node_kind(G, name, "outlet")
            else:
                net.set_node_kind(G, name, "internal")

        if not nx.is_connected(G):
            continue

        from canonical import rescale_to_volume
        rescale_to_volume(G, target_volume)
        return G

    return None


def generate_network(mechanism, seed, target_volume):
    cfg = MECHANISM_CONFIG[mechanism]
    rng = np.random.default_rng(seed)

    n_internal = int(rng.integers(cfg["n_internal_range"][0], cfg["n_internal_range"][1] + 1))
    retain_fraction = float(rng.uniform(*cfg["retain_range"]))

    outlets = outlet_points()
    inlet = np.array([[0.0, 0.0]])
    internal = _place_internal_points(cfg["placement"], rng, n_internal)

    all_points = np.vstack([inlet, outlets, internal])
    node_names = ["in"] + [f"out{i}" for i in range(N_OUTLETS)] + [f"mid{i}" for i in range(len(internal))]
    inlet_idx = 0
    outlet_idxs = set(range(1, 1 + N_OUTLETS))

    G = _build_graph_from_points(all_points, node_names, inlet_idx, outlet_idxs,
                                  retain_fraction, rng, target_volume)
    return G, {"mechanism": mechanism, "seed": seed, "n_internal": n_internal,
               "retain_fraction": retain_fraction}
