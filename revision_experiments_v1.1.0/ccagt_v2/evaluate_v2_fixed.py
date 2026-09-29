#!/usr/bin/env python3
"""Evaluate one frozen matched-representation model on CCAgT v2 without tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MATCHED = PROJECT / "revision_experiments" / "matched_representation"
sys.path.insert(0, str(PROJECT / "baseline"))
sys.path.insert(0, str(MATCHED))

from ccagt_baseline.center_detection import (  # noqa: E402
    extract_local_maxima,
    match_points,
    point_average_precision,
)
from ccagt_baseline.utils import load_config  # noqa: E402
from train_matched_representation import infer_image  # noqa: E402
from train_matched_representation import CenterMultiTaskUNet  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        return
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def point_summary(rows: list[dict], localization_errors: list[float]) -> dict:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    positive = [row for row in rows if int(row["true_count"]) > 0]
    zero = [row for row in rows if int(row["true_count"]) == 0]
    return {
        "point_precision": tp / (tp + fp) if tp + fp else 0.0,
        "point_recall": tp / (tp + fn) if tp + fn else 0.0,
        "point_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_per_image": fp / max(len(rows), 1),
        "count_mae_all": float(
            np.mean([row["absolute_count_error"] for row in rows])
        ),
        "count_mae_positive": (
            float(np.mean([row["absolute_count_error"] for row in positive]))
            if positive
            else None
        ),
        "zero_case_fp": (
            float(np.mean([row["predicted_count"] for row in zero]))
            if zero
            else None
        ),
        "positive_case_recall": (
            sum(int(row["tp"]) for row in positive)
            / max(sum(int(row["tp"]) + int(row["fn"]) for row in positive), 1)
        ),
        "localization_error": (
            float(np.mean(localization_errors)) if localization_errors else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not (HERE / "audit_results.json").is_file():
        raise RuntimeError("v2 audit is incomplete")
    audit = json.loads((HERE / "audit_results.json").read_text(encoding="utf-8"))
    if not audit["evaluation_gate_passed"]:
        raise RuntimeError("Too few nonoverlapping v2 patient/slide units")
    if int(audit["unique_image_ids"]) != 9339:
        raise RuntimeError("v2 image-count gate failed")

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "calibrated_operating_threshold" not in checkpoint:
        raise RuntimeError("Checkpoint has no validation-calibrated threshold")
    model_run = Path(config["output"])
    if not (model_run / "formal_test" / "TEST_COMPLETE_DO_NOT_RERUN").is_file():
        raise RuntimeError("Matched model has not completed its v1 formal test")

    threshold = float(checkpoint["calibrated_operating_threshold"])
    match_radius = float(checkpoint["calibration_match_radius_px"])
    expected_radius = float(config["point_evaluation"]["checkpoint_match_radius_px"])
    if match_radius != expected_radius:
        raise RuntimeError("Checkpoint/config matching-radius mismatch")
    candidate_floor = float(config["point_evaluation"]["ap_candidate_threshold"])
    nms_radius = int(config["point_evaluation"]["nms_radius_px"])
    patch_size = tuple(int(value) for value in config["training"]["patch_size"])
    overlap = float(config["inference"]["overlap"])

    metadata = read_csv(HERE / "metadata.csv")
    eligible_units = set(audit["eligible_patient_slide_units"])
    records_by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata:
        if row["patient_id"] in eligible_units:
            records_by_patient[row["patient_id"]].append(row)
    if set(records_by_patient) != eligible_units:
        raise RuntimeError("Eligible-unit audit/metadata mismatch")
    centers_by_id: dict[str, list[list[float]]] = defaultdict(list)
    for row in read_csv(HERE / "satellite_centers.csv"):
        centers_by_id[row["image_id"]].append(
            [float(row["row"]), float(row["column"])]
        )
    centers = {
        image_id: np.asarray(points, dtype=np.float32).reshape(-1, 2)
        for image_id, points in centers_by_id.items()
    }

    device = torch.device("cuda")
    model = CenterMultiTaskUNet(
        3, base=int(config["model"]["base_channels"]), mask_auxiliary=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": utc_now(),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "sha256": {
            "config": sha256(config_path),
            "checkpoint": sha256(checkpoint_path),
            "protocol": sha256(HERE / "V2_PROTOCOL.md"),
            "audit": sha256(HERE / "audit_results.json"),
        },
        "operating_threshold": threshold,
        "candidate_floor": candidate_floor,
        "nms_radius_px": nms_radius,
        "matching_radius_px": match_radius,
        "eligible_patient_slide_units": sorted(eligible_units),
        "excluded_patient_slide_units": audit["excluded_patient_slide_units"],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    for patient_id in sorted(records_by_patient):
        patient_output = output_root / patient_id
        marker = patient_output / "EVALUATION_COMPLETE_DO_NOT_RERUN"
        if marker.is_file():
            if not (patient_output / "summary.json").is_file():
                raise RuntimeError(f"Completion marker without summary: {patient_output}")
            print(f"skip completed {patient_id}", flush=True)
            continue
        if patient_output.exists():
            raise RuntimeError(f"Refusing incomplete output: {patient_output}")

        image_rows = []
        predicted_rows = []
        localization_errors: list[float] = []
        ap_truth = {}
        ap_predictions = {}
        patient_records = sorted(
            records_by_patient[patient_id], key=lambda row: row["image_id"]
        )
        for index, row in enumerate(patient_records, 1):
            image_id = row["image_id"]
            with Image.open(HERE / row["image_path"]) as source:
                image = np.asarray(source.convert("RGB"))
            _, heatmap = infer_image(model, image, patch_size, overlap, device, 3)
            predicted_points, predicted_scores = extract_local_maxima(
                heatmap, threshold, nms_radius
            )
            candidate_points, candidate_scores = extract_local_maxima(
                heatmap, candidate_floor, nms_radius
            )
            truth = centers.get(image_id, np.empty((0, 2), dtype=np.float32))
            matched = match_points(truth, predicted_points, match_radius)
            localization_errors.extend(matched.localization_errors)
            image_rows.append(
                {
                    "image_id": image_id,
                    "patient_id": patient_id,
                    "tp": matched.tp,
                    "fp": matched.fp,
                    "fn": matched.fn,
                    "true_count": len(truth),
                    "predicted_count": len(predicted_points),
                    "absolute_count_error": abs(len(truth) - len(predicted_points)),
                }
            )
            for point_id, ((point_row, column), score) in enumerate(
                zip(predicted_points, predicted_scores), 1
            ):
                predicted_rows.append(
                    {
                        "image_id": image_id,
                        "point_id": point_id,
                        "row": float(point_row),
                        "column": float(column),
                        "score": float(score),
                    }
                )
            ap_truth[image_id] = truth
            ap_predictions[image_id] = (candidate_points, candidate_scores)
            if index % 100 == 0 or index == len(patient_records):
                print(
                    f"{patient_id}: evaluated {index}/{len(patient_records)}",
                    flush=True,
                )

        summary = {
            "patient_id": patient_id,
            "images": len(patient_records),
            "satellite_instances": sum(len(points) for points in ap_truth.values()),
            "point_ap": point_average_precision(ap_truth, ap_predictions, match_radius),
            **point_summary(image_rows, localization_errors),
            "operating_threshold": threshold,
            "candidate_floor": candidate_floor,
            "nms_radius_px": nms_radius,
            "matching_radius_px": match_radius,
        }
        patient_output.mkdir(parents=True, exist_ok=False)
        write_csv(patient_output / "per_image_metrics.csv", image_rows)
        write_csv(patient_output / "predicted_points.csv", predicted_rows)
        (patient_output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        marker.write_text(utc_now() + "\n", encoding="utf-8")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
