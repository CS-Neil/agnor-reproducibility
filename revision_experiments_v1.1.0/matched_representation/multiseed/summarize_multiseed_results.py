#!/usr/bin/env python3
"""Summarize three-seed matched-formulation held-out results."""

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
OUTPUT = HERE / "summary"
TARGETS = ("mask", "center")
FOLDS = range(3)
SEEDS = (42, 123, 2026)
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
    output = []
    for value in values:
        if value is None or value == "":
            continue
        number = float(value)
        if math.isfinite(number):
            output.append(number)
    return output


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
    rows = []
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
                for required in (summary_path, split_path, checkpoint_path, marker):
                    if not required.is_file():
                        raise RuntimeError(f"Incomplete multiseed input: {required}")
                test_rows = [row for row in read_csv(split_path) if row["split"] == "test"]
                patients = {row["patient_id"] for row in test_rows}
                image_ids = {row["image_id"] for row in test_rows}
                if len(patients) != 1:
                    raise RuntimeError(f"Expected one held-out patient: {name}")
                if fold in split_ids_by_fold and image_ids != split_ids_by_fold[fold]:
                    raise RuntimeError(f"Held-out split mismatch: {name}")
                split_ids_by_fold[fold] = image_ids
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                rows.append(
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
    if len(rows) != 18:
        raise RuntimeError(f"Expected 18 model results, got {len(rows)}")
    write_csv(OUTPUT / "model_results.csv", rows)

    paired = []
    for seed in SEEDS:
        for fold in FOLDS:
            mask = next(
                row for row in rows
                if row["target"] == "mask" and row["fold"] == fold and row["seed"] == seed
            )
            center = next(
                row for row in rows
                if row["target"] == "center" and row["fold"] == fold and row["seed"] == seed
            )
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
    write_csv(OUTPUT / "patient_seed_paired_differences.csv", paired)

    patient_rows = []
    for fold in FOLDS:
        group = [row for row in paired if row["fold"] == fold]
        patient_rows.append(
            {
                "fold": fold,
                "heldout_patient": group[0]["heldout_patient"],
                "seeds": len(group),
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
    write_csv(OUTPUT / "patient_seed_summary.csv", patient_rows)

    cohort_rows = []
    for metric in METRICS:
        patient_values = [
            row[f"center_minus_mask_{metric}_seed_mean"] for row in patient_rows
        ]
        repeated_values = [row[f"center_minus_mask_{metric}"] for row in paired]
        cohort_rows.append(
            {
                "metric": metric,
                "independent_patients": len(finite(patient_values)),
                "seeds_per_patient": len(SEEDS),
                "patient_mean_of_seed_means": mean(patient_values),
                "patient_sd_of_seed_means": sd(patient_values),
                "positive_repeated_pairs": sum(value > 0 for value in finite(repeated_values)),
                "negative_repeated_pairs": sum(value < 0 for value in finite(repeated_values)),
                "finite_repeated_pairs": len(finite(repeated_values)),
            }
        )
    write_csv(OUTPUT / "cohort_summary.csv", cohort_rows)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": digest(HERE / "MULTISEED_PROTOCOL.md"),
        "seeds": list(SEEDS),
        "models": len(rows),
        "paired_patient_seed_rows": len(paired),
        "independent_patients": len(patient_rows),
        "source_hashes": hashes,
        "notes": [
            "Seeds are repeated trained-model measurements, not independent patients.",
            "Cohort means and SDs use three patient-level seed means.",
            "No held-out result selected a checkpoint or operating threshold.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "MULTISEED_SUMMARY_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
