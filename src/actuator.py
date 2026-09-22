from __future__ import annotations

import itertools

import networkx as nx
import numpy as np

import jacobian as jac
import control as ctl
import demand as dem


def eligible_actuator_edges(G: nx.Graph) -> list:
    kinds = dict(G.nodes(data="kind"))
    return [(u, v) for u, v in G.edges() if kinds[u] == "internal" or kinds[v] == "internal"]


def corner_z_samples(K: int, beta: float, n_points: int = 16) -> np.ndarray:
    half = beta / 2.0
    all_corners = np.array(list(itertools.product([-half, half], repeat=K)))
    if n_points >= len(all_corners):
        return all_corners
    idx = np.linspace(0, len(all_corners) - 1, n_points).round().astype(int)
    return all_corners[idx]


def score_placement(G: nx.Graph, candidate: list, outlet_order: list, B: np.ndarray,
                     z_samples: np.ndarray) -> float:
    sigmas = []
    for z in z_samples:
        _, dy_dz = jac.jacobian(G, candidate, outlet_order, z)
        Jc = jac.project_jacobian(dy_dz, B)
        sigmas.append(jac.placement_score(Jc))
    return float(np.quantile(sigmas, 0.1))


def rank_candidates(G: nx.Graph, outlet_order: list, K: int = 4, beta: float = np.log(2.0),
                     n_z_samples: int = 8) -> list:
    eligible = eligible_actuator_edges(G)
    N = len(outlet_order)
    B = jac.zero_sum_basis(N)
    z_samples = corner_z_samples(K, beta, n_z_samples)

    scored = []
    for candidate in itertools.combinations(eligible, K):
        candidate = list(candidate)
        score = score_placement(G, candidate, outlet_order, B, z_samples)
        scored.append((score, candidate))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def eligible_placements(G: nx.Graph, K: int = 4) -> list:
    eligible = sorted(eligible_actuator_edges(G), key=lambda e: (e[0], e[1]))
    return list(itertools.combinations(eligible, K))


def sample_placements(G: nx.Graph, K: int = 4, budget: int = 100, seed: int = 0) -> list:
    all_p = eligible_placements(G, K)
    if len(all_p) <= budget:
        return all_p
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(all_p), size=budget, replace=False))
    return [all_p[i] for i in idx]


def select_placement_nonlinear(G: nx.Graph, outlet_order: list, screening_demands: list,
                                K: int = 4, budget: int = 100, seed: int = 0,
                                beta: float = np.log(2.0), cache: dict | None = None) -> dict:
    placements = sample_placements(G, K, budget, seed)
    evaluated = []
    for p in placements:
        key = tuple(sorted(p))
        if cache is not None and key in cache:
            err = cache[key]
        else:
            err = evaluate_candidate(G, list(p), outlet_order, screening_demands, beta)
            if cache is not None:
                cache[key] = err
        evaluated.append((err, tuple(sorted(p))))

    evaluated.sort(key=lambda t: (t[0], t[1]))
    best_error, best_candidate = evaluated[0]
    return {
        "candidate": list(best_candidate),
        "mean_error": best_error,
        "n_evaluated": len(placements),
        "all": evaluated,
    }


def frozen_screening_demand_set(n_outlets: int = 8, n_directions: int = 10, seed: int = 999999,
                                 severity_fraction: float = 0.5):
    lib = dem.generate_demand_library(n_outlets=n_outlets, n_directions=n_directions, seed=seed)
    b = np.full(n_outlets, 1.0 / n_outlets)
    demands = [dem.demand_at_severity(b, d.v_hat, severity_fraction * d.s_max) for d in lib]
    return demands


def evaluate_candidate(G: nx.Graph, candidate: list, outlet_order: list, screening_demands: list,
                        beta: float, n_starts: int = 6, seed: int = 0) -> float:
    errors = []
    for i, d in enumerate(screening_demands):
        result = ctl.solve_inverse(G, candidate, outlet_order, d, beta, n_starts=n_starts, seed=seed + i)
        errors.append(result["best_error"])
    return float(np.mean(errors))


def select_placements(G: nx.Graph, outlet_order: list, screening_demands: list,
                       K: int = 4, beta: float = np.log(2.0), n_top: int = 5) -> dict:
    scored = rank_candidates(G, outlet_order, K, beta)
    top = scored[:n_top]

    evaluated = []
    for score, candidate in top:
        mean_error = evaluate_candidate(G, candidate, outlet_order, screening_demands, beta)
        evaluated.append({"score": score, "candidate": candidate, "mean_error": mean_error})

    good = min(evaluated, key=lambda r: r["mean_error"])

    n = len(scored)
    poor_idx = int(0.9 * n)
    poor_score, poor_candidate = scored[min(poor_idx, n - 1)]
    poor_mean_error = evaluate_candidate(G, poor_candidate, outlet_order, screening_demands, beta)

    return {
        "n_candidates": n,
        "top_evaluated": evaluated,
        "good": good,
        "poor": {"score": poor_score, "candidate": poor_candidate, "mean_error": poor_mean_error},
    }
