#!/usr/bin/env python3
"""Summarize candidate-floor outcomes at the independent patient/unit level."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "candidate_floor_model_patient.csv"
METRICS = ("point_ap", "maximum_recall", "candidates_per_image")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    numbers = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(numbers)


def sd(values) -> float | None:
    numbers = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.stdev(numbers) if len(numbers) > 1 else None


def main() -> None:
    source = read_csv(SOURCE)
    unit_rows = []
    keys = sorted(
        {
            (
                row["dataset"],
                row["truth_definition"],
                float(row["candidate_floor"]),
                row["patient_id"],
                row["target"],
            )
            for row in source
        }
    )
    for dataset, truth, floor, patient, target in keys:
        rows = [
            row for row in source
            if row["dataset"] == dataset
            and row["truth_definition"] == truth
            and float(row["candidate_floor"]) == floor
            and row["patient_id"] == patient
            and row["target"] == target
        ]
        if dataset == "v1" and len(rows) != 1:
            raise RuntimeError(f"Expected one held-out v1 model: {patient}/{target}")
        if dataset == "v2" and len(rows) != 3:
            raise RuntimeError(f"Expected three fold models: {patient}/{target}")
        unit_rows.append(
            {
                "dataset": dataset,
                "truth_definition": truth,
                "candidate_floor": floor,
                "patient_id": patient,
                "target": target,
                "fold_models": len(rows),
                **{metric: mean(row[metric] for row in rows) for metric in METRICS},
            }
        )
    write_csv(HERE / "candidate_floor_patient_units.csv", unit_rows)

    arm_rows = []
    paired_rows = []
    summary_keys = sorted(
        {
            (row["dataset"], row["truth_definition"], float(row["candidate_floor"]))
            for row in unit_rows
        }
    )
    for dataset, truth, floor in summary_keys:
        selected = [
            row for row in unit_rows
            if row["dataset"] == dataset
            and row["truth_definition"] == truth
            and float(row["candidate_floor"]) == floor
        ]
        for target in ("mask", "center"):
            group = [row for row in selected if row["target"] == target]
            arm_rows.append(
                {
                    "dataset": dataset,
                    "truth_definition": truth,
                    "candidate_floor": floor,
                    "target": target,
                    "independent_patient_units": len(group),
                    **{f"mean_{metric}": mean(row[metric] for row in group) for metric in METRICS},
                    **{f"sample_sd_{metric}": sd(row[metric] for row in group) for metric in METRICS},
                }
            )
        patients = sorted({row["patient_id"] for row in selected})
        differences = {metric: [] for metric in METRICS}
        for patient in patients:
            mask = next(row for row in selected if row["patient_id"] == patient and row["target"] == "mask")
            center = next(row for row in selected if row["patient_id"] == patient and row["target"] == "center")
            for metric in METRICS:
                differences[metric].append(center[metric] - mask[metric])
        paired_rows.append(
            {
                "dataset": dataset,
                "truth_definition": truth,
                "candidate_floor": floor,
                "independent_patient_units": len(patients),
                **{f"mean_center_minus_mask_{metric}": mean(differences[metric]) for metric in METRICS},
                **{f"sample_sd_center_minus_mask_{metric}": sd(differences[metric]) for metric in METRICS},
                **{f"positive_units_{metric}": sum(value > 0 for value in differences[metric]) for metric in METRICS},
            }
        )
    write_csv(HERE / "candidate_floor_arm_summary.csv", arm_rows)
    write_csv(HERE / "candidate_floor_paired_summary_expanded.csv", paired_rows)


if __name__ == "__main__":
    main()
