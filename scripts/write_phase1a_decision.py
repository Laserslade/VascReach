import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def write_decision():
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "phase1a_decision.csv")
    rows = [
        {"item": "total_rays", "value": "240"},
        {"item": "total_severity_points", "value": "9600"},
        {"item": "rays_with_primary_reentry", "value": "2"},
        {"item": "reentry_rays", "value": "symmetric_tree/variant1/direction1; symmetric_tree/variant1/direction3"},
        {"item": "verification_method", "value": "differential_evolution + L-BFGS-B polish"},
        {"item": "verification_points_checked", "value": "6 (deduplicated across shared rho=0 target)"},
        {"item": "verification_agreement", "value": "exact match to displayed precision at all 6 points"},
        {"item": "mechanism", "value": "smooth unimodal (U-shaped) E*(rho) curve crossing "
                                         "epsilon=0.01 twice; jittered network's own baseline "
                                         "output does not match b, so E*(0) sits just above "
                                         "epsilon, dips below it at moderate severity, then "
                                         "rises monotonically thereafter"},
        {"item": "decision", "value": "accessibility is NOT universally interval-like; "
                                        "s_crit is NOT adopted as primary representation; "
                                        "A_G(s;epsilon) remains the primary representation"},
        {"item": "secondary_events_further_verified", "value": "0 (per frozen decision rule, "
                                                                  "not required once primary "
                                                                  "re-entry is confirmed)"},
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "value"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("Saved", out_path)


if __name__ == "__main__":
    write_decision()
