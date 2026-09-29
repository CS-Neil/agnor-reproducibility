from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "formal_lopo_results.csv"
OUTPUT = ROOT / "results" / "formal_lopo_summary_recomputed.csv"
METRICS = (
    "nucleus_dice",
    "nucleus_iou",
    "cluster_dice",
    "cluster_iou",
    "point_ap",
    "point_precision",
    "point_recall",
    "point_f1",
    "fp_per_image",
    "count_mae_all",
    "count_mae_positive",
    "zero_case_fp",
    "positive_case_recall",
    "localization_error",
)


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    methods = ("Standard nnU-Net", "Balanced", "Proposed C4")
    output = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 3:
            raise SystemExit(f"Expected three folds for {method}, got {len(selected)}")
        result = {"method": method}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            result[f"{metric}_mean"] = math.fsum(values) / len(values)
            result[f"{metric}_sd"] = statistics.stdev(values)
        output.append(result)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
