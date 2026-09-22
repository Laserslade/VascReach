import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import demand as dem

TOL = 1e-9


def test_severity_norm_identity():
    positions = dem.outlet_positions_uniform(8)
    delta_x = dem.median_outlet_spacing(positions)
    rng = np.random.default_rng(0)
    d = dem.generate_demand_direction(rng, positions, "regional", delta_x, 0, 0)
    b = np.full(8, 1.0 / 8)

    for s in [0.01, 0.05, d.s_max * 0.5, d.s_max * 0.99]:
        dvec = dem.demand_at_severity(b, d.v_hat, s)
        actual_s = np.linalg.norm(dvec - b, ord=2)
        assert abs(actual_s - s) < TOL
    print("PASS: ||d(s) - b||_2 = s for all tested s")


def test_nonnegativity_up_to_smax():
    positions = dem.outlet_positions_uniform(8)
    delta_x = dem.median_outlet_spacing(positions)
    b = np.full(8, 1.0 / 8)
    rng = np.random.default_rng(1)

    for family in ["localized", "regional", "multifocal"]:
        for i in range(20):
            d = dem.generate_demand_direction(rng, positions, family, delta_x, i, 1)
            for frac in [0.0, 0.25, 0.5, 0.9, 0.999]:
                s = frac * d.s_max
                dvec = dem.demand_at_severity(b, d.v_hat, s)
                assert np.all(dvec >= -1e-12)
    print("PASS: d(s) stays nonnegative for s in [0, s_max)")


def test_reproducibility():
    lib1 = dem.generate_demand_library(n_outlets=8, n_directions=30, seed=42)
    lib2 = dem.generate_demand_library(n_outlets=8, n_directions=30, seed=42)
    for d1, d2 in zip(lib1, lib2):
        assert np.allclose(d1.v_hat, d2.v_hat)
        assert abs(d1.s_max - d2.s_max) < TOL
    print("PASS: same seed reproduces identical demand library")


def test_multifocal_separation():
    positions = dem.outlet_positions_uniform(8)
    delta_x = dem.median_outlet_spacing(positions)
    rng = np.random.default_rng(2)
    for i in range(50):
        d = dem.generate_demand_direction(rng, positions, "multifocal", delta_x, i, 2)
        sep = abs(d.centers[0] - d.centers[1])
        assert sep >= delta_x - 1e-9
    print("PASS: all multifocal centers separated by >= delta_x")


def test_family_balance_and_width_ranges():
    lib = dem.generate_demand_library(n_outlets=8, n_directions=90, seed=7)
    positions = dem.outlet_positions_uniform(8)
    delta_x = dem.median_outlet_spacing(positions)

    counts = {"localized": 0, "regional": 0, "multifocal": 0}
    for d in lib:
        counts[d.pattern_family] += 1
        ratio = d.sigmas[0] / delta_x
        if d.pattern_family == "localized":
            assert 0.5 - 1e-9 <= ratio <= 1.0 + 1e-9
        elif d.pattern_family == "regional":
            assert 1.0 - 1e-9 <= ratio <= 2.0 + 1e-9
        else:
            assert 0.5 - 1e-9 <= ratio <= 2.0 + 1e-9

    assert counts["localized"] == counts["regional"] == counts["multifocal"] == 30
    print(f"PASS: family balance {counts}, width ranges respected")


def run_all():
    tests = [
        test_severity_norm_identity,
        test_nonnegativity_up_to_smax,
        test_reproducibility,
        test_multifocal_separation,
        test_family_balance_and_width_ranges,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"FAIL: {t.__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} tests FAILED.")
        sys.exit(1)
    print(f"All {len(tests)} demand-generator checks PASSED.")


if __name__ == "__main__":
    run_all()
