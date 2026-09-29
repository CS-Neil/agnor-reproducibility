#!/usr/bin/env python3
"""Create prespecified patient-level summaries for fixed-threshold v2 results."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EVALUATIONS = HERE / "evaluations"
METRICS = (
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty summary: {path}")
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_mean(values) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else None


def finite_sd(values) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=1)) if len(array) > 1 else None


def main() -> None:
    audit = json.loads((HERE / "audit_results.json").read_text(encoding="utf-8"))
    if not audit["evaluation_gate_passed"]:
        raise RuntimeError("v2 evaluation gate did not pass")
    eligible_units = sorted(audit["eligible_patient_slide_units"])
    model_rows = []
    for target in ("mask", "center"):
        for fold in range(3):
            model = f"{target}_f{fold}_s42"
            root = EVALUATIONS / model
            for patient_id in eligible_units:
                patient_dir = root / patient_id
                if not (patient_dir / "EVALUATION_COMPLETE_DO_NOT_RERUN").is_file():
                    raise RuntimeError(f"Incomplete evaluation: {patient_dir}")
                summary = json.loads(
                    (patient_dir / "summary.json").read_text(encoding="utf-8")
                )
                model_rows.append(
                    {
                        "target": target,
                        "fold": fold,
                        "model": model,
                        **summary,
                    }
                )
    expected_rows = 6 * len(eligible_units)
    if len(model_rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} model-patient results, got {len(model_rows)}"
        )
    write_csv(HERE / "model_patient_metrics.csv", model_rows)

    arm_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    fold_groups: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
    for row in model_rows:
        arm_groups[(row["target"], row["patient_id"])].append(row)
        fold_groups[(int(row["fold"]), row["patient_id"])][row["target"]] = row

    arm_rows = []
    for (target, patient_id), rows in sorted(arm_groups.items()):
        if len(rows) != 3:
            raise RuntimeError(f"Expected three folds for {target}/{patient_id}")
        arm_rows.append(
            {
                "target": target,
                "patient_id": patient_id,
                "fold_models": 3,
                **{
                    metric: finite_mean(row[metric] for row in rows)
                    for metric in METRICS
                },
            }
        )
    write_csv(HERE / "patient_arm_metrics.csv", arm_rows)

    paired_by_patient: dict[str, list[dict]] = defaultdict(list)
    fold_difference_rows = []
    for (fold, patient_id), arms in sorted(fold_groups.items()):
        if set(arms) != {"mask", "center"}:
            raise RuntimeError(f"Unpaired fold result: fold={fold}, patient={patient_id}")
        row = {
            "fold": fold,
            "patient_id": patient_id,
            **{
                f"center_minus_mask_{metric}": (
                    None
                    if arms["center"][metric] is None or arms["mask"][metric] is None
                    else float(arms["center"][metric]) - float(arms["mask"][metric])
                )
                for metric in METRICS
            },
        }
        fold_difference_rows.append(row)
        paired_by_patient[patient_id].append(row)
    write_csv(HERE / "fold_patient_paired_differences.csv", fold_difference_rows)

    patient_difference_rows = []
    for patient_id, rows in sorted(paired_by_patient.items()):
        if len(rows) != 3:
            raise RuntimeError(f"Expected three paired folds for {patient_id}")
        patient_difference_rows.append(
            {
                "patient_id": patient_id,
                "paired_folds": 3,
                **{
                    f"center_minus_mask_{metric}": finite_mean(
                        row[f"center_minus_mask_{metric}"] for row in rows
                    )
                    for metric in METRICS
                },
            }
        )
    if len(patient_difference_rows) != len(eligible_units):
        raise RuntimeError("Independent patient-level result count is incorrect")
    write_csv(HERE / "patient_paired_differences.csv", patient_difference_rows)

    cohort_rows = []
    for metric in METRICS:
        values = [row[f"center_minus_mask_{metric}"] for row in patient_difference_rows]
        cohort_rows.append(
            {
                "metric": metric,
                "direction": "center_minus_mask",
                "independent_unit": "v2_patient_slide",
                "n": sum(value is not None and np.isfinite(value) for value in values),
                "mean": finite_mean(values),
                "sample_sd": finite_sd(values),
            }
        )
    write_csv(HERE / "cohort_paired_summary.csv", cohort_rows)
    outputs = (
        "model_patient_metrics.csv",
        "patient_arm_metrics.csv",
        "fold_patient_paired_differences.csv",
        "patient_paired_differences.csv",
        "cohort_paired_summary.csv",
    )
    marker = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "eligible_patient_slide_units": eligible_units,
        "model_patient_rows": len(model_rows),
        "sha256": {name: sha256(HERE / name) for name in outputs},
    }
    temporary = (HERE / "V2_SUMMARY_COMPLETE").with_suffix(".tmp")
    temporary.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    temporary.replace(HERE / "V2_SUMMARY_COMPLETE")


if __name__ == "__main__":
    main()
