from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


class HydraulicSolveError(RuntimeError):
    pass


def solve_network(
    G: nx.Graph,
    p_inlet: float = 1.0,
    p_outlet: float = 0.0,
) -> dict:
    nodes = list(G.nodes())
    inlets = [n for n, d in G.nodes(data=True) if d.get("kind") == "inlet"]
    outlets = [n for n, d in G.nodes(data=True) if d.get("kind") == "outlet"]

    if len(inlets) != 1:
        raise HydraulicSolveError(f"Expected exactly 1 inlet, found {len(inlets)}")
    if len(outlets) == 0:
        raise HydraulicSolveError("Network has no outlet nodes")

    fixed = {inlets[0]: p_inlet}
    for o in outlets:
        fixed[o] = p_outlet

    free_nodes = [n for n in nodes if n not in fixed]
    free_idx = {n: k for k, n in enumerate(free_nodes)}
    n_free = len(free_nodes)

    if n_free == 0:
        pressures = dict(fixed)
    else:
        L = lil_matrix((n_free, n_free))
        b = np.zeros(n_free)

        for u, v, data in G.edges(data=True):
            R = data["resistance"]
            if R <= 0:
                raise HydraulicSolveError(f"Non-positive resistance on edge ({u},{v}): {R}")
            g = 1.0 / R

            u_free = u in free_idx
            v_free = v in free_idx

            if u_free:
                L[free_idx[u], free_idx[u]] += g
                if v_free:
                    L[free_idx[u], free_idx[v]] -= g
                else:
                    b[free_idx[u]] += g * fixed[v]

            if v_free:
                L[free_idx[v], free_idx[v]] += g
                if u_free:
                    L[free_idx[v], free_idx[u]] -= g
                else:
                    b[free_idx[v]] += g * fixed[u]

        diag = np.array(L.tocsr().diagonal())
        if np.any(diag == 0):
            bad = [free_nodes[i] for i in np.where(diag == 0)[0]]
            raise HydraulicSolveError(f"Free node(s) {bad} have no conductance path")

        L_csr = csr_matrix(L)
        try:
            P_free = spsolve(L_csr, b)
        except Exception as exc:
            raise HydraulicSolveError(f"Linear solve failed: {exc}") from exc

        pressures = dict(fixed)
        for n, k in free_idx.items():
            pressures[n] = float(P_free[k])

    flows = {}
    for u, v, data in G.edges(data=True):
        R = data["resistance"]
        Q = (pressures[u] - pressures[v]) / R
        flows[(u, v)] = Q

    outlet_flows = {o: 0.0 for o in outlets}
    for (u, v), Q in flows.items():
        if u in outlet_flows:
            outlet_flows[u] += -Q
        if v in outlet_flows:
            outlet_flows[v] += Q

    return {
        "pressures": pressures,
        "flows": flows,
        "outlet_flows": outlet_flows,
    }


def outlet_flow_vector(G: nx.Graph, solve_result: dict, outlet_order: list) -> np.ndarray:
    of = solve_result["outlet_flows"]
    return np.array([of[o] for o in outlet_order], dtype=float)


def normalized_outlet_flow_vector(G: nx.Graph, solve_result: dict, outlet_order: list) -> np.ndarray:
    y = outlet_flow_vector(G, solve_result, outlet_order)
    total = y.sum()
    if total <= 0:
        raise HydraulicSolveError(f"Non-positive total outlet flow ({total})")
    return y / total


def total_dissipation(solve_result: dict, G: nx.Graph) -> float:
    D = 0.0
    for (u, v), Q in solve_result["flows"].items():
        R = G[u][v]["resistance"]
        D += R * Q**2
    return D


def equivalent_resistance(G: nx.Graph, solve_result: dict, p_inlet: float, p_outlet: float) -> float:
    Q_total = sum(solve_result["outlet_flows"].values())
    if Q_total <= 0:
        raise HydraulicSolveError("Non-positive total flow; cannot compute R_eq")
    return (p_inlet - p_outlet) / Q_total
