import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import canonical as can
import actuator as act
from test_canonical import outlet_order_for

BETA = np.log(2.0)


def test_screening_set_frozen_and_reproducible():
    s1 = act.frozen_screening_demand_set()
    s2 = act.frozen_screening_demand_set()
    for d1, d2 in zip(s1, s2):
        assert np.allclose(d1, d2)
    assert len(s1) == 10
    for d in s1:
        assert abs(d.sum() - 1.0) < 1e-9
        assert np.all(d >= -1e-12)
    print("PASS: screening demand set is frozen, reproducible, and valid")


def test_top5_scores_at_least_as_good_as_median():
    families, _ = can.build_all_families()
    G = families["symmetric_tree"]
    outlet_order = outlet_order_for(G)
    scored = act.rank_candidates(G, outlet_order)
    scores = [s for s, _ in scored]
    top5_min = min(scores[:5])
    median = np.median(scores)
    assert top5_min >= median, f"Top-5 should score at or above the median: {top5_min} vs {median}"
    print(f"PASS: top-5 Jacobian scores ({[f'{s:.4f}' for s in scores[:5]]}) all >= median ({median:.4f})")


def test_poor_placement_scores_below_good():
    families, _ = can.build_all_families()
    G = families["symmetric_tree"]
    outlet_order = outlet_order_for(G)
    screening = act.frozen_screening_demand_set()
    result = act.select_placements(G, outlet_order, screening)

    good_score = next(r["score"] for r in result["top_evaluated"] if r["candidate"] == result["good"]["candidate"])
    poor_score = result["poor"]["score"]
    assert poor_score < good_score, f"Poor placement should score below good: {poor_score} vs {good_score}"
    print(f"PASS: poor placement score ({poor_score:.4f}) below selected good placement score ({good_score:.4f})")


def run_full_pipeline_all_families():
    families, target_volume = can.build_all_families()
    screening = act.frozen_screening_demand_set()

    results = {}
    for name, G in families.items():
        outlet_order = outlet_order_for(G)
        t0 = time.time()
        r = act.select_placements(G, outlet_order, screening)
        elapsed = time.time() - t0
        results[name] = r
        print(f"{name}: n_candidates={r['n_candidates']}, "
              f"good_score={r['good']['score']:.4f}, good_mean_error={r['good']['mean_error']:.2e}, "
              f"poor_score={r['poor']['score']:.4f}, poor_mean_error={r['poor']['mean_error']:.2e}, "
              f"time={elapsed:.1f}s")
        if r["poor"]["mean_error"] < r["good"]["mean_error"]:
            print(f"  FLAG: for {name}, the Jacobian-screened placement underperformed "
                  f"the deliberately poor one in full nonlinear evaluation. "
                  f"The linear proxy did not predict nonlinear accessibility here.")

        scores = [e["score"] for e in r["top_evaluated"]] + [r["poor"]["score"]]
        errs = [e["mean_error"] for e in r["top_evaluated"]] + [r["poor"]["mean_error"]]
        rho = float(np.corrcoef(scores, errs)[0, 1])
        print(f"  score-vs-error correlation across evaluated candidates: {rho:.3f} "
              f"(negative expected: higher score -> lower error)")
    return results


def run_all():
    tests = [
        test_screening_set_frozen_and_reproducible,
        test_top5_scores_at_least_as_good_as_median,
        test_poor_placement_scores_below_good,
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
    print(f"All {len(tests)} actuator-pipeline checks PASSED.")
    print()
    print("Full pipeline across canonical families:")
    run_full_pipeline_all_families()


if __name__ == "__main__":
    run_all()
