from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DemandDirection:
    v_hat: np.ndarray
    s_max: float
    pattern_family: str
    centers: np.ndarray
    sigmas: np.ndarray
    amplitudes: np.ndarray
    seed: int
    direction_id: int


def outlet_positions_uniform(n_outlets: int, domain_length: float = 1.0) -> np.ndarray:
    return np.linspace(0.0, domain_length, n_outlets)


def median_outlet_spacing(positions: np.ndarray) -> float:
    diffs = np.diff(np.sort(positions))
    return float(np.median(diffs))


def _sample_centers(rng: np.random.Generator, domain_length: float, k: int,
                     min_separation: float | None = None) -> np.ndarray:
    if k == 1:
        return rng.uniform(0.0, domain_length, size=1)
    if k == 2 and min_separation is not None:
        for _ in range(1000):
            c = rng.uniform(0.0, domain_length, size=2)
            if abs(c[0] - c[1]) >= min_separation:
                return c
        raise RuntimeError("Could not sample separated multifocal centers")
    return rng.uniform(0.0, domain_length, size=k)


def generate_demand_direction(
    rng: np.random.Generator,
    positions: np.ndarray,
    pattern_family: str,
    delta_x: float,
    direction_id: int,
    seed: int,
    domain_length: float = 1.0,
) -> DemandDirection:
    n = len(positions)
    b = np.full(n, 1.0 / n)

    if pattern_family == "localized":
        k = 1
        log_ratio = rng.uniform(np.log(0.5), np.log(1.0))
    elif pattern_family == "regional":
        k = 1
        log_ratio = rng.uniform(np.log(1.0), np.log(2.0))
    elif pattern_family == "multifocal":
        k = 2
        log_ratio = rng.uniform(np.log(0.5), np.log(2.0))
    else:
        raise ValueError(f"Unknown pattern family: {pattern_family}")

    sigma = np.exp(log_ratio) * delta_x
    sigmas = np.full(k, sigma)
    centers = _sample_centers(rng, domain_length, k, min_separation=delta_x if k == 2 else None)

    if k == 1:
        amplitudes = np.array([1.0])
    else:
        ratio = rng.uniform(0.5, 1.5)
        amplitudes = np.array([1.0, ratio])

    a = np.zeros(n)
    for A_k, c_k, sig_k in zip(amplitudes, centers, sigmas):
        a += A_k * np.exp(-((positions - c_k) ** 2) / (2.0 * sig_k**2))

    v = a - a.mean()
    norm = np.linalg.norm(v, ord=2)
    if norm < 1e-14:
        raise RuntimeError("Degenerate activation field, resample")
    v_hat = v / norm
    s_max = _analytic_s_max(b, v_hat)

    return DemandDirection(
        v_hat=v_hat,
        s_max=s_max,
        pattern_family=pattern_family,
        centers=centers,
        sigmas=sigmas,
        amplitudes=amplitudes,
        seed=seed,
        direction_id=direction_id,
    )


def _analytic_s_max(b: np.ndarray, v_hat: np.ndarray) -> float:
    negative_mask = v_hat < 0
    if not np.any(negative_mask):
        raise RuntimeError("v_hat has no negative components, s_max undefined")
    candidates = -b[negative_mask] / v_hat[negative_mask]
    return float(np.min(candidates))


def demand_at_severity(b: np.ndarray, v_hat: np.ndarray, s: float) -> np.ndarray:
    return b + s * v_hat


def generate_demand_library(
    n_outlets: int,
    n_directions: int,
    seed: int,
    families: tuple = ("localized", "regional", "multifocal"),
) -> list[DemandDirection]:
    positions = outlet_positions_uniform(n_outlets)
    delta_x = median_outlet_spacing(positions)

    master_rng = np.random.default_rng(seed)
    directions = []
    for i in range(n_directions):
        family = families[i % len(families)]
        sub_seed = int(master_rng.integers(0, 2**31 - 1))
        sub_rng = np.random.default_rng(sub_seed)
        d = generate_demand_direction(
            sub_rng, positions, family, delta_x, direction_id=i, seed=sub_seed
        )
        directions.append(d)
    return directions
