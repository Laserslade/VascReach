from __future__ import annotations

import networkx as nx
import numpy as np


DEFAULT_MU = 1.0e-3


def new_network(mu: float = DEFAULT_MU) -> nx.Graph:
    G = nx.Graph()
    G.graph["mu"] = mu
    return G


def add_channel(
    G: nx.Graph,
    u,
    v,
    length: float,
    radius: float,
    controllable: bool = False,
) -> None:
    if length <= 0:
        raise ValueError(f"Channel length must be positive, got {length}")
    if radius <= 0:
        raise ValueError(f"Channel radius must be positive, got {radius}")

    mu = G.graph["mu"]
    R = hagen_poiseuille_resistance(mu, length, radius)

    G.add_edge(
        u,
        v,
        length=length,
        radius=radius,
        resistance=R,
        resistance0=R,
        controllable=controllable,
    )


def hagen_poiseuille_resistance(mu: float, length: float, radius: float) -> float:
    return 8.0 * mu * length / (np.pi * radius**4)


def rectangular_channel_resistance(
    mu: float, length: float, width: float, height: float, n_terms: int = 20
) -> float:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    w, h = max(width, height), min(width, height)

    n = np.arange(1, 2 * n_terms, 2)
    series_sum = np.sum((192.0 / (np.pi**5)) * (h / w) / n**5 * np.tanh(n * np.pi * w / (2 * h)))
    correction = 1.0 - series_sum
    R = 12.0 * mu * length / (w * h**3) / correction
    return float(R)


def set_node_kind(G: nx.Graph, node, kind: str) -> None:
    if kind not in ("inlet", "outlet", "internal"):
        raise ValueError(f"Unknown node kind: {kind}")
    G.nodes[node]["kind"] = kind


def inlet_nodes(G: nx.Graph) -> list:
    return [n for n, d in G.nodes(data=True) if d.get("kind") == "inlet"]


def outlet_nodes(G: nx.Graph) -> list:
    return [n for n, d in G.nodes(data=True) if d.get("kind") == "outlet"]


def controllable_edges(G: nx.Graph) -> list:
    return [(u, v) for u, v, d in G.edges(data=True) if d.get("controllable")]


def total_channel_volume(G: nx.Graph) -> float:
    V = 0.0
    for _, _, d in G.edges(data=True):
        if "width" in d and "height" in d:
            A = d["width"] * d["height"]
        else:
            A = np.pi * d["radius"] ** 2
        V += d["length"] * A
    return V
