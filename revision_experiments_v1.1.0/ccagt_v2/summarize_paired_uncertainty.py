#!/usr/bin/env python3
"""Summarize patient-paired v2 differences with descriptive bootstrap intervals."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE / "patient_paired_differences.csv"
OUTPUT = HERE / "patient_paired_uncertainty.csv"
MARKER = HERE / "PAIRED_UNCERTAINTY_COMPLETE"
SEED = 20260927
RESAMPLES = 100_000
METRICS = (
    "point_ap",
    "point_f1",
    "fp_per_image",
    "count_mae_positive",
    "localization_error",
)


def percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    if len(source_rows) != 12:
        raise SystemExit(f"Expected 12 independent units, found {len(source_rows)}")

    rng = random.Random(SEED)
    result_rows: list[dict[str, str | int]] = []
    for metric in METRICS:
        field = f"center_minus_mask_{metric}"
        values = [float(row[field]) for row in source_rows]
        bootstrap_means = sorted(
            statistics.fmean(rng.choice(values) for _ in values)
            for _ in range(RESAMPLES)
        )
        ordered = sorted(values)
        result_rows.append(
            {
                "metric": metric,
                "independent_units": len(values),
                "mean": f"{statistics.fmean(values):.12g}",
                "median": f"{statistics.median(values):.12g}",
                "q1": f"{percentile(ordered, 0.25):.12g}",
                "q3": f"{percentile(ordered, 0.75):.12g}",
                "bootstrap_mean_ci95_low": f"{percentile(bootstrap_means, 0.025):.12g}",
                "bootstrap_mean_ci95_high": f"{percentile(bootstrap_means, 0.975):.12g}",
                "bootstrap_resamples": RESAMPLES,
                "bootstrap_seed": SEED,
            }
        )

    fieldnames = list(result_rows[0])
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    marker = {
        "status": "complete",
        "input": INPUT.name,
        "input_sha256": sha256(INPUT),
        "output": OUTPUT.name,
        "output_sha256": sha256(OUTPUT),
        "independent_units": 12,
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": SEED,
        "interval": "percentile bootstrap interval for the paired-unit mean",
        "interpretation": "descriptive; no population-level hypothesis test",
    }
    MARKER.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
