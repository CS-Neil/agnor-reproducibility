#!/usr/bin/env python3
"""Render immutable experiment summaries into reviewable manuscript snippets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
MATCHED = HERE / "matched_representation" / "summary"
V2 = HERE / "ccagt_v2"
OUTPUT = HERE / "result_snippets"
MARKER = OUTPUT / "SNIPPETS_COMPLETE"
DISPLAY_METRICS = (
    ("point_ap", "Point AP"),
    ("point_f1", "Point F1"),
    ("fp_per_image", "FP/image"),
    ("count_mae_positive", "MAE+"),
)
SOURCES = (
    MATCHED / "formal_test_results.csv",
    MATCHED / "arm_summary.csv",
    MATCHED / "paired_patient_differences.csv",
    MATCHED / "paired_summary.csv",
    V2 / "model_patient_metrics.csv",
    V2 / "patient_arm_metrics.csv",
    V2 / "patient_paired_differences.csv",
    V2 / "cohort_paired_summary.csv",
    V2 / "V2_SUMMARY_COMPLETE",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: str | float | None, digits: int = 3) -> str:
    if value in (None, ""):
        return "NA"
    numeric = float(value)
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "NA"


def patient_label(value: str) -> str:
    return value.removeprefix("patient_")


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    required_markers = (
        MATCHED / "SUMMARY_COMPLETE",
        V2 / "V2_SUMMARY_COMPLETE",
    )
    for required in (*required_markers, *SOURCES):
        if not required.is_file():
            raise RuntimeError(f"Results are incomplete: {required}")

    source_hashes = {str(path.relative_to(HERE)): sha256(path) for path in SOURCES}
    if MARKER.is_file():
        existing = json.loads(MARKER.read_text(encoding="utf-8"))
        if existing.get("source_sha256") == source_hashes:
            print("Result snippets already complete and source hashes match")
            return
        raise RuntimeError("Refusing to replace snippets generated from different sources")

    matched_results = read_csv(MATCHED / "formal_test_results.csv")
    matched_arms = read_csv(MATCHED / "arm_summary.csv")
    matched_differences = read_csv(MATCHED / "paired_patient_differences.csv")
    matched_paired_summary = {
        row["metric"]: row for row in read_csv(MATCHED / "paired_summary.csv")
    }
    v2_arms = read_csv(V2 / "patient_arm_metrics.csv")
    v2_differences = {
        row["patient_id"]: row
        for row in read_csv(V2 / "patient_paired_differences.csv")
    }
    v2_cohort = {
        row["metric"]: row for row in read_csv(V2 / "cohort_paired_summary.csv")
    }

    if len(matched_results) != 6 or len(matched_arms) != 2:
        raise RuntimeError("Matched summary does not contain six runs and two arms")
    if len(v2_arms) != 24 or len(v2_differences) != 12:
        raise RuntimeError("v2 summary does not contain 12 patients and two arms")

    matched_by_patient = sorted(
        matched_results,
        key=lambda row: (patient_label(row["heldout_patient"]), row["target"]),
    )
    matched_tex = [
        "% Generated from immutable matched-formulation summary files.",
        "% Patient & Arm & AP & F1 & FP/image & MAE+ \\",
    ]
    for row in matched_by_patient:
        matched_tex.append(
            " & ".join(
                (
                    patient_label(row["heldout_patient"]),
                    row["target"].capitalize(),
                    number(row["point_ap"]),
                    number(row["point_f1"]),
                    number(row["fp_per_image"]),
                    number(row["count_mae_positive"]),
                )
            )
            + r" \\" 
        )
    for row in sorted(matched_arms, key=lambda value: value["target"]):
        values = []
        for metric, _ in DISPLAY_METRICS:
            values.append(
                f"${number(row[f'{metric}_mean'])}\\pm"
                f"{number(row[f'{metric}_sample_sd'])}$"
            )
        matched_tex.append(
            " & ".join((f"{row['target'].capitalize()} mean", "--", *values))
            + r" \\" 
        )

    arm_by_key = {(row["target"], row["patient_id"]): row for row in v2_arms}
    v2_tex = [
        "% Generated from immutable CCAgT v2 patient-level summaries.",
        "% Unit & Mask AP & Center AP & Delta AP & Mask F1 & Center F1 & Delta F1 \\",
    ]
    for patient_id in sorted(v2_differences):
        mask = arm_by_key[("mask", patient_id)]
        center = arm_by_key[("center", patient_id)]
        difference = v2_differences[patient_id]
        v2_tex.append(
            " & ".join(
                (
                    patient_label(patient_id),
                    number(mask["point_ap"]),
                    number(center["point_ap"]),
                    number(difference["center_minus_mask_point_ap"]),
                    number(mask["point_f1"]),
                    number(center["point_f1"]),
                    number(difference["center_minus_mask_point_f1"]),
                )
            )
            + r" \\" 
        )

    report = [
        "# Immutable revision-experiment evidence",
        "",
        "This file is generated only after all six matched held-out tests and all",
        "72 model-by-v2-patient evaluations have immutable completion markers.",
        "Differences are center minus mask. Values are descriptive.",
        "",
        "## Matched-formulation arm summary",
        "",
        "| Arm | Independent patients | AP | F1 | FP/image | MAE+ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(matched_arms, key=lambda value: value["target"]):
        values = [
            f"{number(row[f'{metric}_mean'])} ± {number(row[f'{metric}_sample_sd'])}"
            for metric, _ in DISPLAY_METRICS
        ]
        report.append(
            "| " + " | ".join((row["target"], row["independent_heldout_patients"], *values)) + " |"
        )

    report.extend(
        (
            "",
            "## Matched center-minus-mask paired summary",
            "",
            "| Metric | n | Mean difference | Sample SD |",
            "|---|---:|---:|---:|",
        )
    )
    for metric, label in DISPLAY_METRICS:
        row = matched_paired_summary[metric]
        report.append(
            f"| {label} | {row['n']} | {number(row['mean'])} | {number(row['sample_sd'])} |"
        )

    report.extend(
        (
            "",
            "## CCAgT v2 center-minus-mask patient-level cohort summary",
            "",
            "| Metric | Independent units | Mean difference | Sample SD |",
            "|---|---:|---:|---:|",
        )
    )
    for metric, label in DISPLAY_METRICS:
        row = v2_cohort[metric]
        report.append(
            f"| {label} | {row['n']} | {number(row['mean'])} | {number(row['sample_sd'])} |"
        )

    report.extend(
        (
            "",
            "## CCAgT v2 patient-level arm means and paired differences",
            "",
            "| Unit | Mask AP | Center AP | Δ AP | Mask F1 | Center F1 | Δ F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for patient_id in sorted(v2_differences):
        mask = arm_by_key[("mask", patient_id)]
        center = arm_by_key[("center", patient_id)]
        difference = v2_differences[patient_id]
        report.append(
            "| "
            + " | ".join(
                (
                    patient_label(patient_id),
                    number(mask["point_ap"]),
                    number(center["point_ap"]),
                    number(difference["center_minus_mask_point_ap"]),
                    number(mask["point_f1"]),
                    number(center["point_f1"]),
                    number(difference["center_minus_mask_point_f1"]),
                )
            )
            + " |"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write(OUTPUT / "matched_table_rows.tex", "\n".join(matched_tex) + "\n")
    atomic_write(OUTPUT / "v2_table_rows.tex", "\n".join(v2_tex) + "\n")
    atomic_write(OUTPUT / "RESULTS_EVIDENCE.md", "\n".join(report) + "\n")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes,
        "row_counts": {
            "matched_runs": len(matched_results),
            "matched_arms": len(matched_arms),
            "v2_patient_arms": len(v2_arms),
            "v2_patient_differences": len(v2_differences),
        },
        "output_sha256": {
            name: sha256(OUTPUT / name)
            for name in (
                "matched_table_rows.tex",
                "v2_table_rows.tex",
                "RESULTS_EVIDENCE.md",
            )
        },
    }
    atomic_write(OUTPUT / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    atomic_write(
        MARKER,
        json.dumps(
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": source_hashes,
                "manifest_sha256": sha256(OUTPUT / "manifest.json"),
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
