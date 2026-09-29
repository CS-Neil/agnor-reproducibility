#!/usr/bin/env python3
"""Re-run frozen inference for candidate-floor and point-truth sensitivity."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
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
V2 = PROJECT / "revision_experiments" / "ccagt_v2"
POINT_TRUTH = V2 / "point_truth_compatibility"
MODEL_OUTPUT = HERE / "model_results"
FLOORS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
TARGETS = ("mask", "center")
FOLDS = range(3)

sys.path.insert(0, str(PROJECT / "baseline"))
sys.path.insert(0, str(MATCHED))

from ccagt_baseline.center_detection import (  # noqa: E402
    extract_local_maxima,
    match_points,
    point_average_precision,
)
from ccagt_baseline.data import load_pair, resolve_records  # noqa: E402
from ccagt_baseline.utils import load_config  # noqa: E402
from train_matched_representation import (  # noqa: E402
    CenterMultiTaskUNet,
    infer_image,
    load_centers,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        raise RuntimeError(f"Refusing empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def centers_from_csv(path: Path) -> dict[str, np.ndarray]:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[row["image_id"]].append([float(row["row"]), float(row["column"])])
    return {
        image_id: np.asarray(points, dtype=np.float32).reshape(-1, 2)
        for image_id, points in grouped.items()
    }


def fixed_threshold_summary(
    truths: dict[str, np.ndarray],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    match_radius: float,
) -> dict:
    tp = fp = fn = 0
    errors = []
    all_count_errors = []
    positive_count_errors = []
    zero_predictions = []
    positive_tp = positive_truth = 0
    for image_id, truth in truths.items():
        points, _ = predictions[image_id]
        matched = match_points(truth, points, match_radius)
        tp += matched.tp
        fp += matched.fp
        fn += matched.fn
        errors.extend(matched.localization_errors)
        count_error = abs(len(truth) - len(points))
        all_count_errors.append(count_error)
        if len(truth):
            positive_count_errors.append(count_error)
            positive_tp += matched.tp
            positive_truth += len(truth)
        else:
            zero_predictions.append(len(points))
    return {
        "point_precision": tp / (tp + fp) if tp + fp else 0.0,
        "point_recall": tp / (tp + fn) if tp + fn else 0.0,
        "point_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_per_image": fp / max(len(truths), 1),
        "count_mae_all": statistics.fmean(all_count_errors),
        "count_mae_positive": (
            statistics.fmean(positive_count_errors) if positive_count_errors else None
        ),
        "zero_case_fp": statistics.fmean(zero_predictions) if zero_predictions else None,
        "positive_case_recall": positive_tp / positive_truth if positive_truth else 0.0,
        "localization_error": statistics.fmean(errors) if errors else None,
    }


def evaluate_patient(
    *,
    model,
    records,
    image_loader,
    truth_definitions: dict[str, dict[str, np.ndarray]],
    patient_id: str,
    dataset: str,
    target: str,
    fold: int,
    patch_size,
    overlap: float,
    device,
    nms_radius: int,
    match_radius: float,
    operating_threshold: float,
) -> tuple[list[dict], list[dict]]:
    predictions_by_floor = {floor: {} for floor in FLOORS}
    for index, record in enumerate(records, 1):
        image_id, image = image_loader(record)
        _, heatmap = infer_image(model, image, patch_size, overlap, device, 3)
        points, scores = extract_local_maxima(heatmap, FLOORS[0], nms_radius)
        for floor in FLOORS:
            selected = scores >= floor
            predictions_by_floor[floor][image_id] = (points[selected], scores[selected])
        if index % 100 == 0 or index == len(records):
            print(
                f"{target}_f{fold}: {dataset}/{patient_id} {index}/{len(records)}",
                flush=True,
            )

    floor_rows = []
    threshold_rows = []
    for truth_name, truths in truth_definitions.items():
        expected_ids = set(predictions_by_floor[FLOORS[0]])
        patient_truth = {
            image_id: truths.get(image_id, np.empty((0, 2), dtype=np.float32))
            for image_id in expected_ids
        }
        truth_count = sum(len(points) for points in patient_truth.values())
        for floor in FLOORS:
            predictions = predictions_by_floor[floor]
            tp = fn = candidates = 0
            for image_id, truth in patient_truth.items():
                points, _ = predictions[image_id]
                matched = match_points(truth, points, match_radius)
                tp += matched.tp
                fn += matched.fn
                candidates += len(points)
            floor_rows.append(
                {
                    "dataset": dataset,
                    "patient_id": patient_id,
                    "target": target,
                    "fold": fold,
                    "truth_definition": truth_name,
                    "candidate_floor": floor,
                    "images": len(patient_truth),
                    "truth_instances": truth_count,
                    "candidate_count": candidates,
                    "candidates_per_image": candidates / max(len(patient_truth), 1),
                    "maximum_recall": tp / (tp + fn) if tp + fn else 0.0,
                    "point_ap": point_average_precision(
                        patient_truth, predictions, match_radius
                    ),
                }
            )
        operating_predictions = {
            image_id: (
                points[scores >= operating_threshold],
                scores[scores >= operating_threshold],
            )
            for image_id, (points, scores) in predictions_by_floor[FLOORS[0]].items()
        }
        threshold_rows.append(
            {
                "dataset": dataset,
                "patient_id": patient_id,
                "target": target,
                "fold": fold,
                "truth_definition": truth_name,
                "operating_threshold": operating_threshold,
                "point_ap_at_floor_0_05": next(
                    row["point_ap"]
                    for row in floor_rows
                    if row["truth_definition"] == truth_name
                    and row["candidate_floor"] == 0.05
                ),
                **fixed_threshold_summary(
                    patient_truth, operating_predictions, match_radius
                ),
            }
        )
    return floor_rows, threshold_rows


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


def consolidate() -> None:
    all_floor_rows = []
    all_threshold_rows = []
    for target in TARGETS:
        for fold in FOLDS:
            name = f"{target}_f{fold}_s42"
            root = MODEL_OUTPUT / name
            if not (root / "MODEL_ANALYSIS_COMPLETE").is_file():
                raise RuntimeError(f"Incomplete model analysis: {name}")
            all_floor_rows.extend(read_csv(root / "candidate_floor_results.csv"))
            all_threshold_rows.extend(read_csv(root / "truth_definition_results.csv"))
    write_csv(HERE / "candidate_floor_model_patient.csv", all_floor_rows)
    write_csv(HERE / "truth_definition_model_patient.csv", all_threshold_rows)

    summaries = []
    for dataset, truth_name in [("v1", "component_centroid"), ("v2", "bbox_center"), ("v2", "polygon_centroid")]:
        for floor in FLOORS:
            selected = [
                row for row in all_floor_rows
                if row["dataset"] == dataset
                and row["truth_definition"] == truth_name
                and abs(float(row["candidate_floor"]) - floor) < 1e-9
            ]
            patient_differences = []
            patients = sorted({row["patient_id"] for row in selected})
            for patient in patients:
                values = []
                for fold in FOLDS:
                    mask_rows = [
                        row for row in selected
                        if row["patient_id"] == patient
                        and row["target"] == "mask"
                        and int(row["fold"]) == fold
                    ]
                    center_rows = [
                        row for row in selected
                        if row["patient_id"] == patient
                        and row["target"] == "center"
                        and int(row["fold"]) == fold
                    ]
                    if dataset == "v1" and (not mask_rows or not center_rows):
                        continue
                    if mask_rows and center_rows:
                        values.append(
                            float(center_rows[0]["point_ap"])
                            - float(mask_rows[0]["point_ap"])
                        )
                if values:
                    patient_differences.append(statistics.fmean(values))
            summaries.append(
                {
                    "dataset": dataset,
                    "truth_definition": truth_name,
                    "candidate_floor": floor,
                    "independent_patient_units": len(patient_differences),
                    "mean_center_minus_mask_ap": mean(patient_differences),
                    "sample_sd_center_minus_mask_ap": sd(patient_differences),
                    "positive_patient_units": sum(value > 0 for value in patient_differences),
                    "negative_patient_units": sum(value < 0 for value in patient_differences),
                }
            )
    write_csv(HERE / "candidate_floor_summary.csv", summaries)

    comparison_rows = []
    v2_rows = [row for row in all_threshold_rows if row["dataset"] == "v2"]
    metrics = (
        "point_ap_at_floor_0_05",
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
    for target in TARGETS:
        for fold in FOLDS:
            for patient in sorted({row["patient_id"] for row in v2_rows}):
                bbox = next(
                    row for row in v2_rows
                    if row["target"] == target and int(row["fold"]) == fold
                    and row["patient_id"] == patient and row["truth_definition"] == "bbox_center"
                )
                polygon = next(
                    row for row in v2_rows
                    if row["target"] == target and int(row["fold"]) == fold
                    and row["patient_id"] == patient and row["truth_definition"] == "polygon_centroid"
                )
                comparison_rows.append(
                    {
                        "target": target,
                        "fold": fold,
                        "patient_id": patient,
                        **{
                            f"polygon_minus_bbox_{metric}": (
                                None
                                if polygon[metric] == "" or bbox[metric] == ""
                                else float(polygon[metric]) - float(bbox[metric])
                            )
                            for metric in metrics
                        },
                    }
                )
    write_csv(HERE / "truth_definition_differences.csv", comparison_rows)
    truth_summary = []
    for metric in metrics:
        values = finite(
            row[f"polygon_minus_bbox_{metric}"] for row in comparison_rows
        )
        truth_summary.append(
            {
                "metric": metric,
                "model_patient_rows": len(values),
                "mean_polygon_minus_bbox": mean(values),
                "sample_sd_polygon_minus_bbox": sd(values),
                "max_absolute_difference": max(map(abs, values)) if values else None,
                "nonzero_rows": sum(abs(value) > 1e-12 for value in values),
            }
        )
    write_csv(HERE / "truth_definition_summary.csv", truth_summary)

    manifest = {
        "created_at_utc": utc_now(),
        "protocol_sha256": digest(HERE / "PROTOCOL.md"),
        "floors": list(FLOORS),
        "models": 6,
        "v1_heldout_model_patient_rows": sum(
            row["dataset"] == "v1" and abs(float(row["candidate_floor"]) - 0.05) < 1e-9
            for row in all_floor_rows
        ),
        "v2_model_patient_rows": sum(
            row["dataset"] == "v2" and row["truth_definition"] == "bbox_center"
            and abs(float(row["candidate_floor"]) - 0.05) < 1e-9
            for row in all_floor_rows
        ),
        "model_result_hashes": {
            path.parent.name: digest(path)
            for path in sorted(MODEL_OUTPUT.glob("*/candidate_floor_results.csv"))
        },
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "CANDIDATE_FLOOR_ANALYSIS_COMPLETE").write_text(
        utc_now() + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def main() -> None:
    audit = json.loads((V2 / "audit_results.json").read_text(encoding="utf-8"))
    eligible = set(audit["eligible_patient_slide_units"])
    metadata = read_csv(V2 / "metadata.csv")
    v2_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata:
        if row["patient_id"] in eligible:
            v2_records[row["patient_id"]].append(row)
    bbox_centers = centers_from_csv(V2 / "satellite_centers.csv")
    polygon_centers = centers_from_csv(POINT_TRUTH / "polygon_centers.csv")
    v1_centers = load_centers(
        PROJECT / "experiments" / "satellite_center_labels" / "centers.csv"
    )

    device = torch.device("cuda")
    for target in TARGETS:
        for fold in FOLDS:
            name = f"{target}_f{fold}_s42"
            destination = MODEL_OUTPUT / name
            if (destination / "MODEL_ANALYSIS_COMPLETE").is_file():
                print(f"skip completed {name}", flush=True)
                continue
            if destination.exists():
                raise RuntimeError(f"Refusing incomplete output: {destination}")
            config_path = MATCHED / "configs" / f"{name}.yaml"
            config = load_config(config_path)
            checkpoint_path = MATCHED / "runs" / name / "checkpoint_best.pth"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            operating_threshold = float(checkpoint["calibrated_operating_threshold"])
            match_radius = float(checkpoint["calibration_match_radius_px"])
            nms_radius = int(config["point_evaluation"]["nms_radius_px"])
            patch_size = tuple(int(value) for value in config["training"]["patch_size"])
            overlap = float(config["inference"]["overlap"])
            model = CenterMultiTaskUNet(
                3, base=int(config["model"]["base_channels"]), mask_auxiliary=False
            ).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()

            floor_rows = []
            threshold_rows = []
            test_records = [
                record for record in resolve_records(config) if record.role == "test"
            ]
            patient_ids = {record.patient_id for record in test_records}
            if len(patient_ids) != 1:
                raise RuntimeError(f"Expected one v1 test patient: {name}")
            values = evaluate_patient(
                model=model,
                records=test_records,
                image_loader=lambda record: (record.image_id, load_pair(record)[0]),
                truth_definitions={"component_centroid": v1_centers},
                patient_id=next(iter(patient_ids)),
                dataset="v1",
                target=target,
                fold=fold,
                patch_size=patch_size,
                overlap=overlap,
                device=device,
                nms_radius=nms_radius,
                match_radius=match_radius,
                operating_threshold=operating_threshold,
            )
            floor_rows.extend(values[0])
            threshold_rows.extend(values[1])

            def load_v2(row):
                with Image.open(V2 / row["image_path"]) as source:
                    return row["image_id"], np.asarray(source.convert("RGB"))

            for patient_id in sorted(v2_records):
                values = evaluate_patient(
                    model=model,
                    records=sorted(v2_records[patient_id], key=lambda row: row["image_id"]),
                    image_loader=load_v2,
                    truth_definitions={
                        "bbox_center": bbox_centers,
                        "polygon_centroid": polygon_centers,
                    },
                    patient_id=patient_id,
                    dataset="v2",
                    target=target,
                    fold=fold,
                    patch_size=patch_size,
                    overlap=overlap,
                    device=device,
                    nms_radius=nms_radius,
                    match_radius=match_radius,
                    operating_threshold=operating_threshold,
                )
                floor_rows.extend(values[0])
                threshold_rows.extend(values[1])
            destination.mkdir(parents=True, exist_ok=False)
            write_csv(destination / "candidate_floor_results.csv", floor_rows)
            write_csv(destination / "truth_definition_results.csv", threshold_rows)
            (destination / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at_utc": utc_now(),
                        "model": name,
                        "config_sha256": digest(config_path),
                        "checkpoint_sha256": digest(checkpoint_path),
                        "protocol_sha256": digest(HERE / "PROTOCOL.md"),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (destination / "MODEL_ANALYSIS_COMPLETE").write_text(
                utc_now() + "\n", encoding="utf-8"
            )
            del model
            torch.cuda.empty_cache()
    consolidate()


if __name__ == "__main__":
    main()
