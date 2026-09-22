from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
from scipy.linalg import null_space


def resistances_dict(G: nx.Graph, controllable_edges: list, z: np.ndarray) -> dict:
    z_map = dict(zip(controllable_edges, z))
    R = {}
    for u, v, data in G.edges(data=True):
        key = (u, v)
        if key in z_map:
            R[key] = data["resistance0"] * np.exp(z_map[key])
        elif (v, u) in z_map:
            R[key] = data["resistance0"] * np.exp(z_map[(v, u)])
        else:
            R[key] = data["resistance"]
    return R


def _build_free_system(G: nx.Graph, edge_R: dict, p_inlet: float, p_outlet: float):
    inlets = [n for n, d in G.nodes(data=True) if d.get("kind") == "inlet"]
    outlets = [n for n, d in G.nodes(data=True) if d.get("kind") == "outlet"]
    fixed = {inlets[0]: p_inlet}
    for o in outlets:
        fixed[o] = p_outlet

    free_nodes = [n for n in G.nodes() if n not in fixed]
    free_idx = {n: k for k, n in enumerate(free_nodes)}
    n_free = len(free_nodes)

    L = lil_matrix((n_free, n_free))
    b = np.zeros(n_free)
    for u, v in G.edges():
        g = 1.0 / edge_R[(u, v)]
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

    return L.tocsr(), b, free_idx, fixed, outlets


def solve_with_edge_R(G: nx.Graph, edge_R: dict, p_inlet: float = 1.0, p_outlet: float = 0.0):
    L, b, free_idx, fixed, outlets = _build_free_system(G, edge_R, p_inlet, p_outlet)

    if len(free_idx) == 0:
        P_free = np.array([])
        lu = None
    else:
        lu = splu(L.tocsc())
        P_free = lu.solve(b)

    pressures = dict(fixed)
    for n, k in free_idx.items():
        pressures[n] = float(P_free[k])

    flows = {}
    for u, v in G.edges():
        flows[(u, v)] = (pressures[u] - pressures[v]) / edge_R[(u, v)]

    outlet_flows = {o: 0.0 for o in outlets}
    for (u, v), Q in flows.items():
        if u in outlet_flows:
            outlet_flows[u] += -Q
        if v in outlet_flows:
            outlet_flows[v] += Q

    return pressures, flows, outlet_flows, free_idx, fixed, lu


def forward_normalized_flow(G: nx.Graph, controllable_edges: list, outlet_order: list,
                             z: np.ndarray, p_inlet: float = 1.0, p_outlet: float = 0.0) -> np.ndarray:
    edge_R = resistances_dict(G, controllable_edges, z)
    _, _, outlet_flows, _, _, _ = solve_with_edge_R(G, edge_R, p_inlet, p_outlet)
    y_raw = np.array([outlet_flows[o] for o in outlet_order])
    return y_raw / y_raw.sum()


def jacobian(G: nx.Graph, controllable_edges: list, outlet_order: list, z: np.ndarray,
             p_inlet: float = 1.0, p_outlet: float = 0.0):
    edge_R = resistances_dict(G, controllable_edges, z)
    pressures, flows, outlet_flows, free_idx, fixed, lu = solve_with_edge_R(
        G, edge_R, p_inlet, p_outlet
    )

    N = len(outlet_order)
    K = len(controllable_edges)
    y_raw = np.array([outlet_flows[o] for o in outlet_order])
    S = y_raw.sum()
    y = y_raw / S

    n_free = len(free_idx)
    P_free_vec = np.zeros(n_free)
    for n, idx in free_idx.items():
        P_free_vec[idx] = pressures[n]

    dy_raw_dz = np.zeros((N, K))

    for k, (eu, ev) in enumerate(controllable_edges):
        g = 1.0 / edge_R[(eu, ev)]
        dg = -g

        u_free = eu in free_idx
        v_free = ev in free_idx
        dL_entries = {}
        db_local = np.zeros(n_free)

        if u_free:
            iu = free_idx[eu]
            dL_entries[(iu, iu)] = dL_entries.get((iu, iu), 0.0) + dg
            if v_free:
                iv = free_idx[ev]
                dL_entries[(iu, iv)] = dL_entries.get((iu, iv), 0.0) - dg
            else:
                db_local[iu] += dg * fixed[ev]
        if v_free:
            iv = free_idx[ev]
            dL_entries[(iv, iv)] = dL_entries.get((iv, iv), 0.0) + dg
            if u_free:
                iu = free_idx[eu]
                dL_entries[(iv, iu)] = dL_entries.get((iv, iu), 0.0) - dg
            else:
                db_local[iv] += dg * fixed[eu]

        if n_free == 0:
            dP_free = np.array([])
        else:
            dLP = np.zeros(n_free)
            for (r, c), val in dL_entries.items():
                dLP[r] += val * P_free_vec[c]
            dP_free = lu.solve(db_local - dLP)

        dP = {n: 0.0 for n in fixed}
        for n, idx in free_idx.items():
            dP[n] = dP_free[idx]

        dy_raw_map = {o: 0.0 for o in outlet_order}
        for u, v in G.edges():
            R = edge_R[(u, v)]
            dQ_uv = (dP[u] - dP[v]) / R
            if (u, v) == (eu, ev):
                dQ_uv += (pressures[u] - pressures[v]) * dg
            if u in dy_raw_map:
                dy_raw_map[u] += -dQ_uv
            if v in dy_raw_map:
                dy_raw_map[v] += dQ_uv

        dy_raw_dz[:, k] = np.array([dy_raw_map[o] for o in outlet_order])

    dS_dz = dy_raw_dz.sum(axis=0)
    dy_dz = (dy_raw_dz * S - np.outer(y_raw, dS_dz)) / S**2

    return y, dy_dz


def zero_sum_basis(N: int) -> np.ndarray:
    B = null_space(np.ones((1, N)))
    return B.T


def project_jacobian(dy_dz: np.ndarray, B: np.ndarray) -> np.ndarray:
    return B @ dy_dz


def placement_score(Jc: np.ndarray) -> float:
    sv = np.linalg.svd(Jc, compute_uv=False)
    sv_nonzero = sv[sv > 1e-12 * sv.max()]
    return float(sv_nonzero.min()) if len(sv_nonzero) > 0 else 0.0
