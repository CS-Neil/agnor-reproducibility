#!/usr/bin/env python3
"""Summarize available multiseed results without concealing the failed run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
MATCHED = HERE.parent
OUTPUT = HERE / "available_summary"
TARGETS = ("mask", "center")
FOLDS = range(3)
SEEDS = (42, 123, 2026)
FAILED_RUN = "center_f2_s123"
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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing empty summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(values) -> list[float]:
    numbers = []
    for value in values:
        if value is None or value == "":
            continue
        number = float(value)
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def mean(values) -> float | None:
    numbers = finite(values)
    return statistics.fmean(numbers) if numbers else None


def sd(values) -> float | None:
    numbers = finite(values)
    return statistics.stdev(numbers) if len(numbers) > 1 else None


def run_directory(target: str, fold: int, seed: int) -> Path:
    name = f"{target}_f{fold}_s{seed}"
    return (MATCHED / "runs" / name) if seed == 42 else (HERE / "runs" / name)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_rows = []
    availability = []
    hashes = {}
    split_ids_by_fold: dict[int, set[str]] = {}

    for seed in SEEDS:
        for fold in FOLDS:
            for target in TARGETS:
                name = f"{target}_f{fold}_s{seed}"
                run = run_directory(target, fold, seed)
                formal = run / "formal_test"
                summary_path = formal / "test_summary.json"
                split_path = formal / "resolved_split.csv"
                checkpoint_path = run / "checkpoint_best.pth"
                marker = formal / "TEST_COMPLETE_DO_NOT_RERUN"
                complete = all(
                    path.is_file()
                    for path in (summary_path, split_path, checkpoint_path, marker)
                )
                status = "completed" if complete else "missing"
                reason = ""
                if name == FAILED_RUN and not complete:
                    status = "reproducible_nonfinite_gradient_failure"
                    reason = "two attempts failed at epoch 7, same minibatch and parameter"
                availability.append(
                    {
                        "run": name,
                        "target": target,
                        "fold": fold,
                        "seed": seed,
                        "status": status,
                        "reason": reason,
                    }
                )
                if not complete:
                    if name != FAILED_RUN:
                        raise RuntimeError(f"Unexpected incomplete run: {name}")
                    continue

                test_rows = [row for row in read_csv(split_path) if row["split"] == "test"]
                patients = {row["patient_id"] for row in test_rows}
                image_ids = {row["image_id"] for row in test_rows}
                if len(patients) != 1:
                    raise RuntimeError(f"Expected one held-out patient: {name}")
                if fold in split_ids_by_fold and image_ids != split_ids_by_fold[fold]:
                    raise RuntimeError(f"Held-out split mismatch: {name}")
                split_ids_by_fold[fold] = image_ids
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                model_rows.append(
                    {
                        "target": target,
                        "fold": fold,
                        "seed": seed,
                        "heldout_patient": next(iter(patients)),
                        "images": int(summary["images"]),
                        "checkpoint_epoch": int(summary["checkpoint_epoch"]),
                        "operating_threshold": float(summary["operating_threshold"]),
                        **{metric: summary.get(metric) for metric in METRICS},
                    }
                )
                hashes[name] = {
                    "test_summary": digest(summary_path),
                    "resolved_split": digest(split_path),
                    "checkpoint": digest(checkpoint_path),
                }

    if len(model_rows) != 17:
        raise RuntimeError(f"Expected 17 completed model results, got {len(model_rows)}")
    write_csv(OUTPUT / "availability.csv", availability)
    write_csv(OUTPUT / "model_results_available.csv", model_rows)

    paired = []
    for seed in SEEDS:
        for fold in FOLDS:
            mask = next(
                (row for row in model_rows if row["target"] == "mask" and row["fold"] == fold and row["seed"] == seed),
                None,
            )
            center = next(
                (row for row in model_rows if row["target"] == "center" and row["fold"] == fold and row["seed"] == seed),
                None,
            )
            if mask is None or center is None:
                continue
            if mask["heldout_patient"] != center["heldout_patient"]:
                raise RuntimeError("Unpaired held-out patients")
            paired.append(
                {
                    "fold": fold,
                    "seed": seed,
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
    if len(paired) != 8:
        raise RuntimeError(f"Expected 8 available pairs, got {len(paired)}")
    write_csv(OUTPUT / "paired_differences_available.csv", paired)

    patient_rows = []
    for fold in FOLDS:
        group = [row for row in paired if row["fold"] == fold]
        patient_rows.append(
            {
                "fold": fold,
                "heldout_patient": group[0]["heldout_patient"],
                "available_seeds": len(group),
                "seed_ids": ";".join(str(row["seed"]) for row in group),
                **{
                    f"center_minus_mask_{metric}_seed_mean": mean(
                        row[f"center_minus_mask_{metric}"] for row in group
                    )
                    for metric in METRICS
                },
                **{
                    f"center_minus_mask_{metric}_seed_sd": sd(
                        row[f"center_minus_mask_{metric}"] for row in group
                    )
                    for metric in METRICS
                },
            }
        )
    write_csv(OUTPUT / "patient_available_seed_summary.csv", patient_rows)

    seed_rows = []
    for seed in SEEDS:
        group = [row for row in paired if row["seed"] == seed]
        for metric in METRICS:
            values = finite(row[f"center_minus_mask_{metric}"] for row in group)
            seed_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "available_independent_patients": len(values),
                    "mean_center_minus_mask": mean(values),
                    "sample_sd_center_minus_mask": sd(values),
                    "positive_patients": sum(value > 0 for value in values),
                    "negative_patients": sum(value < 0 for value in values),
                }
            )
    write_csv(OUTPUT / "seed_specific_summary.csv", seed_rows)

    descriptive_rows = []
    for metric in METRICS:
        values = finite(row[f"center_minus_mask_{metric}"] for row in paired)
        patient_means = finite(
            row[f"center_minus_mask_{metric}_seed_mean"] for row in patient_rows
        )
        descriptive_rows.append(
            {
                "metric": metric,
                "available_patient_seed_pairs": len(values),
                "mean_across_available_pairs_descriptive": mean(values),
                "sample_sd_across_available_pairs_descriptive": sd(values),
                "positive_available_pairs": sum(value > 0 for value in values),
                "negative_available_pairs": sum(value < 0 for value in values),
                "unbalanced_patient_mean": mean(patient_means),
                "unbalanced_patient_sd": sd(patient_means),
            }
        )
    write_csv(OUTPUT / "descriptive_available_summary.csv", descriptive_rows)

    failure_hashes = {
        path.parent.name: digest(path)
        for path in sorted((HERE / "failed_attempts").glob("*/training.log"))
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": digest(HERE / "MULTISEED_PROTOCOL.md"),
        "retry_log_sha256": digest(HERE / "RETRY_LOG.md"),
        "completed_models_including_seed42": len(model_rows),
        "planned_models_including_seed42": 18,
        "available_pairs": len(paired),
        "planned_pairs": 9,
        "failed_run": FAILED_RUN,
        "source_hashes": hashes,
        "failed_attempt_training_log_hashes": failure_hashes,
        "interpretation": (
            "The prespecified three-seed protocol is not fully complete. "
            "Available-pair summaries are descriptive and explicitly unbalanced."
        ),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "AVAILABLE_SUMMARY_COMPLETE_NOT_FULL_PROTOCOL").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
