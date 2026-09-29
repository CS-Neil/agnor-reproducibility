#!/usr/bin/env python3
"""Summarize the immutable held-out results for the matched target ablation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUTPUT = HERE / "summary"
TARGETS = ("mask", "center")
FOLDS = range(3)
METRICS = (
    "nucleus_dice",
    "cluster_dice",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty summary: {path}")
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_values(values) -> list[float]:
    output = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            output.append(numeric)
    return output


def mean(values) -> float | None:
    numbers = finite_values(values)
    return statistics.fmean(numbers) if numbers else None


def sample_sd(values) -> float | None:
    numbers = finite_values(values)
    return statistics.stdev(numbers) if len(numbers) > 1 else None


def main() -> None:
    model_rows = []
    source_hashes = {}
    test_ids_by_fold: dict[int, set[str]] = {}
    for target in TARGETS:
        for fold in FOLDS:
            name = f"{target}_f{fold}_s42"
            run = RUNS / name
            formal = run / "formal_test"
            marker = formal / "TEST_COMPLETE_DO_NOT_RERUN"
            summary_path = formal / "test_summary.json"
            split_path = formal / "resolved_split.csv"
            checkpoint_path = run / "checkpoint_best.pth"
            for required in (marker, summary_path, split_path, checkpoint_path):
                if not required.is_file():
                    raise RuntimeError(f"Incomplete formal result: {required}")

            test_rows = [row for row in read_csv(split_path) if row["split"] == "test"]
            patient_ids = {row["patient_id"] for row in test_rows}
            if len(patient_ids) != 1:
                raise RuntimeError(f"Expected one held-out patient for {name}")
            image_ids = {row["image_id"] for row in test_rows}
            if fold in test_ids_by_fold and image_ids != test_ids_by_fold[fold]:
                raise RuntimeError(f"Mask/center held-out images differ for fold {fold}")
            test_ids_by_fold[fold] = image_ids

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["split"] != "test" or int(summary["images"]) != len(image_ids):
                raise RuntimeError(f"Test summary/split mismatch for {name}")
            model_rows.append(
                {
                    "target": target,
                    "fold": fold,
                    "heldout_patient": next(iter(patient_ids)),
                    "images": int(summary["images"]),
                    "checkpoint_epoch": int(summary["checkpoint_epoch"]),
                    "operating_threshold": float(summary["operating_threshold"]),
                    "matching_radius_px": float(summary["matching_radius_px"]),
                    **{metric: summary[metric] for metric in METRICS},
                }
            )
            source_hashes[name] = {
                "test_summary": sha256(summary_path),
                "resolved_split": sha256(split_path),
                "checkpoint": sha256(checkpoint_path),
            }

    if len(model_rows) != 6:
        raise RuntimeError(f"Expected six formal results, got {len(model_rows)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "formal_test_results.csv", model_rows)

    by_target = {
        target: sorted(
            (row for row in model_rows if row["target"] == target),
            key=lambda row: int(row["fold"]),
        )
        for target in TARGETS
    }
    arm_rows = []
    for target, rows in by_target.items():
        arm_rows.append(
            {
                "target": target,
                "independent_heldout_patients": len(rows),
                **{
                    f"{metric}_mean": mean(row[metric] for row in rows)
                    for metric in METRICS
                },
                **{
                    f"{metric}_sample_sd": sample_sd(
                        row[metric] for row in rows
                    )
                    for metric in METRICS
                },
            }
        )
    write_csv(OUTPUT / "arm_summary.csv", arm_rows)

    paired_rows = []
    for fold in FOLDS:
        mask = next(row for row in by_target["mask"] if int(row["fold"]) == fold)
        center = next(
            row for row in by_target["center"] if int(row["fold"]) == fold
        )
        if mask["heldout_patient"] != center["heldout_patient"]:
            raise RuntimeError(f"Held-out patients are unpaired for fold {fold}")
        paired_rows.append(
            {
                "fold": fold,
                "heldout_patient": mask["heldout_patient"],
                **{
                    f"center_minus_mask_{metric}": (
                        None
                        if center[metric] is None or mask[metric] is None
                        else float(center[metric]) - float(mask[metric])
                    )
                    for metric in METRICS
                },
            }
        )
    write_csv(OUTPUT / "paired_patient_differences.csv", paired_rows)

    paired_summary = []
    for metric in METRICS:
        values = [row[f"center_minus_mask_{metric}"] for row in paired_rows]
        finite = finite_values(values)
        paired_summary.append(
            {
                "metric": metric,
                "direction": "center_minus_mask",
                "independent_unit": "heldout_patient",
                "n": len(finite),
                "mean": mean(finite),
                "sample_sd": sample_sd(finite),
            }
        )
    write_csv(OUTPUT / "paired_summary.csv", paired_summary)

    (OUTPUT / "manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_hashes": source_hashes,
                "notes": [
                    "All values are descriptive because there are three independent held-out patients.",
                    "Differences are center minus mask and are paired within fold/patient.",
                    "No checkpoint or threshold is selected from these held-out results.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "SUMMARY_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
