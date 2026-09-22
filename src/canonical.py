from __future__ import annotations

import networkx as nx
import numpy as np

import network as net

NOMINAL_RADIUS = 5e-5
NOMINAL_LENGTH = 1e-3


def _binary_tree_edges(prefix, depth):
    if depth == 0:
        return [], [prefix]
    left, right = prefix + "0", prefix + "1"
    e_l, leaves_l = _binary_tree_edges(left, depth - 1)
    e_r, leaves_r = _binary_tree_edges(right, depth - 1)
    edges = [(prefix, left), (prefix, right)] + e_l + e_r
    return edges, leaves_l + leaves_r


def _node_depth(name):
    return len(name) - len("in")


def _finalize(G, leaves, n_outlets):
    mapping = {leaf: f"out{i}" for i, leaf in enumerate(leaves)}
    G = nx.relabel_nodes(G, mapping)
    net.set_node_kind(G, "in", "inlet")
    for i in range(n_outlets):
        net.set_node_kind(G, f"out{i}", "outlet")
    for n in G.nodes():
        if n != "in" and not n.startswith("out"):
            net.set_node_kind(G, n, "internal")
    return G


def symmetric_tree(n_outlets: int = 8) -> nx.Graph:
    depth = int(round(np.log2(n_outlets)))
    edges, leaves = _binary_tree_edges("in", depth)
    G = net.new_network()
    for u, v in edges:
        net.add_channel(G, u, v, length=NOMINAL_LENGTH, radius=NOMINAL_RADIUS)
    return _finalize(G, leaves, n_outlets)


def murray_tree(n_outlets: int = 8, leaf_radius: float = NOMINAL_RADIUS) -> nx.Graph:
    depth = int(round(np.log2(n_outlets)))
    edges, leaves = _binary_tree_edges("in", depth)
    G = net.new_network()
    for u, v in edges:
        d = _node_depth(v)
        radius = leaf_radius * 2.0 ** ((depth - d) / 3.0)
        net.add_channel(G, u, v, length=NOMINAL_LENGTH, radius=radius)
    return _finalize(G, leaves, n_outlets)


def looped_hierarchical(n_outlets: int = 8) -> nx.Graph:
    depth = int(round(np.log2(n_outlets)))
    edges, leaves = _binary_tree_edges("in", depth)
    G = net.new_network()
    for u, v in edges:
        net.add_channel(G, u, v, length=NOMINAL_LENGTH, radius=NOMINAL_RADIUS)

    all_nodes = set()
    for u, v in edges:
        all_nodes.add(u)
        all_nodes.add(v)
    for parent in all_nodes:
        c0, c1 = parent + "0", parent + "1"
        if c0 in all_nodes and c1 in all_nodes and _node_depth(c0) < depth:
            if not G.has_edge(c0, c1):
                net.add_channel(G, c0, c1, length=NOMINAL_LENGTH, radius=NOMINAL_RADIUS)

    return _finalize(G, leaves, n_outlets)


def grid_lattice(n_outlets: int = 8, size: int = 5) -> nx.Graph:
    if n_outlets != 8 or size != 5:
        raise ValueError("grid_lattice is currently frozen to the 5x5/8-outlet Phase 1 case")

    raw = nx.grid_2d_graph(size, size)
    G = net.new_network()
    for (u, v) in raw.edges():
        net.add_channel(G, u, v, length=NOMINAL_LENGTH, radius=NOMINAL_RADIUS)

    center = (2, 2)
    outlet_nodes = [(0, 1), (0, 3), (1, 0), (1, 4), (3, 0), (3, 4), (4, 1), (4, 3)]

    def angle(node):
        i, j = node
        return np.arctan2(i - 2, j - 2)

    outlet_nodes.sort(key=angle)

    mapping = {center: "in"}
    for i, n in enumerate(outlet_nodes):
        mapping[n] = f"out{i}"
    remaining = [n for n in raw.nodes() if n not in mapping]
    for i, n in enumerate(remaining):
        mapping[n] = f"mid{i}"

    G = nx.relabel_nodes(G, mapping)
    net.set_node_kind(G, "in", "inlet")
    for i in range(n_outlets):
        net.set_node_kind(G, f"out{i}", "outlet")
    for n in G.nodes():
        if n != "in" and not n.startswith("out"):
            net.set_node_kind(G, n, "internal")

    return G


FAMILY_BUILDERS = {
    "symmetric_tree": symmetric_tree,
    "murray_tree": murray_tree,
    "looped_hierarchical": looped_hierarchical,
    "grid_lattice": grid_lattice,
}


def rescale_to_volume(G: nx.Graph, target_volume: float) -> nx.Graph:
    current = net.total_channel_volume(G)
    factor = np.sqrt(target_volume / current)
    for u, v, data in G.edges(data=True):
        new_radius = data["radius"] * factor
        new_R = net.hagen_poiseuille_resistance(G.graph["mu"], data["length"], new_radius)
        data["radius"] = new_radius
        data["resistance"] = new_R
        data["resistance0"] = new_R
    return G


FAMILY_VARIANT_OFFSET = {
    "symmetric_tree": 0,
    "murray_tree": 100000,
    "looped_hierarchical": 200000,
    "grid_lattice": 300000,
}


def build_family_variant(family_name: str, variant_seed: int, target_volume: float,
                          jitter_frac: float = 0.15) -> nx.Graph:
    G = FAMILY_BUILDERS[family_name]()
    if variant_seed != 0:
        rng = np.random.default_rng(
            2_000_000 + variant_seed * 100000 + FAMILY_VARIANT_OFFSET[family_name]
        )
        for u, v, data in G.edges(data=True):
            factor = 1.0 + rng.uniform(-jitter_frac, jitter_frac)
            data["length"] = data["length"] * factor
            data["resistance"] = net.hagen_poiseuille_resistance(
                G.graph["mu"], data["length"], data["radius"]
            )
            data["resistance0"] = data["resistance"]
    rescale_to_volume(G, target_volume)
    return G


def build_pilot_networks(n_variants: int = 3, target_volume: float | None = None) -> dict:
    base, tv = build_all_families()
    if target_volume is None:
        target_volume = tv
    pilots = {}
    for family in FAMILY_BUILDERS:
        for variant in range(n_variants):
            G = build_family_variant(family, variant, target_volume)
            pilots[(family, variant)] = G
    return pilots, target_volume


def build_all_families(n_outlets: int = 8, target_volume: float | None = None) -> dict:
    raw = {name: builder(n_outlets) for name, builder in FAMILY_BUILDERS.items()}
    if target_volume is None:
        target_volume = net.total_channel_volume(raw["symmetric_tree"])
    for G in raw.values():
        rescale_to_volume(G, target_volume)
    return raw, target_volume
