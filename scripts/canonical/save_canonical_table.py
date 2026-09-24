import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

from test_canonical import build_validation_table


def save_csv(outpath):
    rows, target_volume = build_validation_table()
    fields = ["family", "n_nodes", "n_edges", "connected", "n_inlets", "n_outlets",
              "volume", "R_eq", "dissipation", "E_baseline"]
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})
    print("Saved", outpath, "target_volume=", target_volume)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "canonical")
    os.makedirs(out_dir, exist_ok=True)
    save_csv(os.path.join(out_dir, "canonical_networks.csv"))
