from __future__ import annotations

import numpy as np

import demand as dem

TERCILE_EDGES = [(0.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0)]
PATTERN_CYCLE = ["localized", "regional", "multifocal"]


def generate_calibration_problems(n_per_stratum: int = 20, n_outlets: int = 8, seed: int = 555555):
    positions = dem.outlet_positions_uniform(n_outlets)
    delta_x = dem.median_outlet_spacing(positions)
    master_rng = np.random.default_rng(seed)

    problems = []
    pid = 0
    for tercile, (lo, hi) in enumerate(TERCILE_EDGES):
        for i in range(n_per_stratum):
            family = PATTERN_CYCLE[i % len(PATTERN_CYCLE)]
            sub_seed = int(master_rng.integers(0, 2**31 - 1))
            sub_rng = np.random.default_rng(sub_seed)
            direction = dem.generate_demand_direction(
                sub_rng, positions, family, delta_x, direction_id=pid, seed=sub_seed
            )
            rho = float(master_rng.uniform(lo, hi))
            s = rho * direction.s_max
            problems.append({
                "pid": pid,
                "tercile": tercile,
                "pattern_family": family,
                "rho": rho,
                "s": s,
                "v_hat": direction.v_hat,
                "seed": sub_seed,
            })
            pid += 1
    return problems
