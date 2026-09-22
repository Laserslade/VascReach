from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.optimize import minimize

import jacobian as jac


class ControlError(RuntimeError):
    pass


def _raw_outlet_flows(G, controllable_edges, outlet_order, z):
    edge_R = jac.resistances_dict(G, controllable_edges, z)
    _, _, outlet_flows, _, _, _ = jac.solve_with_edge_R(G, edge_R)
    y_raw = np.array([outlet_flows[o] for o in outlet_order])
    S = y_raw.sum()
    scale = np.abs(y_raw).max()
    if scale == 0.0 or abs(S) < 1e-8 * scale:
        raise ControlError(f"Total outlet flow ({S}) negligible relative to flow scale ({scale})")
    return y_raw, S


def objective_and_grad(z, G, controllable_edges, outlet_order, d):
    _raw_outlet_flows(G, controllable_edges, outlet_order, z)
    y, dy_dz = jac.jacobian(G, controllable_edges, outlet_order, z)
    N = len(d)
    diff = y - d
    L = float(np.dot(diff, diff) / N)
    grad = (2.0 / N) * (dy_dz.T @ diff)
    return L, grad


def tracking_error(G, controllable_edges, outlet_order, z, d):
    y, _ = jac.jacobian(G, controllable_edges, outlet_order, z)
    N = len(d)
    return float(np.sqrt(np.dot(y - d, y - d) / N))


def solve_inverse(
    G: nx.Graph,
    controllable_edges: list,
    outlet_order: list,
    d: np.ndarray,
    beta: float,
    n_starts: int = 8,
    seed: int = 0,
    warm_start_z: np.ndarray | None = None,
) -> dict:
    K = len(controllable_edges)
    bounds = [(-beta, beta)] * K
    rng = np.random.default_rng(seed)

    starts = [np.zeros(K)]
    if warm_start_z is not None:
        starts.append(np.clip(np.asarray(warm_start_z, dtype=float), -beta, beta))
    while len(starts) < n_starts:
        starts.append(rng.uniform(-beta, beta, size=K))

    per_start = []
    for z0 in starts:
        res = minimize(
            objective_and_grad,
            z0,
            args=(G, controllable_edges, outlet_order, d),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-16, "gtol": 1e-14, "maxiter": 1000},
        )
        E = float(np.sqrt(max(res.fun, 0.0)))
        per_start.append({
            "z0": z0,
            "z": res.x,
            "error": E,
            "success": bool(res.success),
            "n_iter": int(res.nit),
        })

    best = min(per_start, key=lambda r: r["error"])
    y_raw, S = _raw_outlet_flows(G, controllable_edges, outlet_order, best["z"])
    y_hat = y_raw / S

    return {
        "best_z": best["z"],
        "best_error": best["error"],
        "best_outlet_flows": y_raw,
        "best_normalized_flows": y_hat,
        "control_effort": float(np.linalg.norm(best["z"])),
        "optimizer_success": best["success"],
        "n_starts": len(starts),
        "per_start": per_start,
    }
