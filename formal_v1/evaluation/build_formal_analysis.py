#!/usr/bin/env python3
"""Package frozen LOPO results and build the final descriptive analyses."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT = Path(os.environ.get("CANCER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
SOURCE = Path(os.environ.get("FORMAL_ARTIFACT_ROOT", "/private/tmp/cancer_formal_transfer/extracted"))
RELEASE = PROJECT / "releases/formal_lopo_final"
TABLES = PROJECT / "paper/tables"
FIGURES = PROJECT / "paper/figures"
METHODS = ("Standard nnU-Net", "Balanced", "Proposed C4")
PATIENTS = {0: "Patient A", 1: "Patient B", 2: "Patient C"}
TRAIN = {0: "Patient B", 1: "Patient C", 2: "Patient A"}
VALIDATION = {0: "Patient C", 1: "Patient A", 2: "Patient B"}
SUMMARY_KEYS = (
    "nucleus_dice", "nucleus_iou", "cluster_dice", "cluster_iou",
    "point_ap", "point_precision", "point_recall", "point_f1",
    "fp_per_image", "count_mae_all", "count_mae_positive", "zero_case_fp",
    "positive_case_recall", "localization_error",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_path(method: str, fold: int) -> Path:
    if method == "Proposed C4":
        return SOURCE / f"experiments/final/C4_w020_centerFP32_final/formal_test/fold{fold}/seed42/test_summary.json"
    slug = "standard" if method == "Standard nnU-Net" else "balanced"
    return SOURCE / f"experiments/final/nnunet_{slug}_lopo/fold{fold}/seed42/formal_test/evaluation/test_summary.json"


def candidate_path(method: str, fold: int) -> Path:
    if method == "Proposed C4":
        return SOURCE / f"experiments/final/C4_w020_centerFP32_final/formal_test/fold{fold}/seed42/predicted_centers.csv"
    slug = "standard" if method == "Standard nnU-Net" else "balanced"
    return SOURCE / f"experiments/final/nnunet_{slug}_lopo/fold{fold}/seed42/formal_test/evaluation/test_candidate_points.csv"


def threshold(method: str, fold: int) -> float:
    return float(json.loads(summary_path(method, fold).read_text())["operating_threshold"])


def load_centers() -> dict[str, np.ndarray]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in read_csv(SOURCE / "experiments/satellite_center_labels/centers.csv"):
        grouped[row["image_id"]].append((float(row["row"]), float(row["column"])))
    return {key: np.asarray(value, dtype=np.float32) for key, value in grouped.items()}


def load_candidates(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[row["image_id"]].append(
            (float(row["row"]), float(row["column"]), float(row["score"]))
        )
    result = {}
    for image_id, values in grouped.items():
        values.sort(key=lambda value: -value[2])
        array = np.asarray(values, dtype=np.float32)
        result[image_id] = (array[:, :2], array[:, 2])
    return result


def pr_curve(truth: dict[str, np.ndarray], predictions: dict[str, tuple[np.ndarray, np.ndarray]], radius: float = 5.0):
    total_true = sum(len(value) for value in truth.values())
    detections = []
    for image_id, (points, scores) in predictions.items():
        detections.extend((float(score), image_id, index, point) for index, (point, score) in enumerate(zip(points, scores)))
    detections.sort(key=lambda value: (-value[0], value[1], value[2]))
    used = {key: np.zeros(len(value), dtype=bool) for key, value in truth.items()}
    tp, fp = [], []
    for _, image_id, _, point in detections:
        targets = truth.get(image_id, np.empty((0, 2), dtype=np.float32))
        matched = used.setdefault(image_id, np.zeros(len(targets), dtype=bool))
        hit = False
        if len(targets):
            distances = np.linalg.norm(targets - point[None], axis=1)
            distances[matched] = np.inf
            index = int(np.argmin(distances))
            hit = bool(distances[index] <= radius)
            if hit:
                matched[index] = True
        tp.append(float(hit)); fp.append(float(not hit))
    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    recall = tp_c / max(total_true, 1)
    precision = tp_c / np.maximum(tp_c + fp_c, 1)
    return recall, precision


def match_counts(truth_points: np.ndarray, predicted_points: np.ndarray, radius: float = 5.0):
    if not len(truth_points) or not len(predicted_points):
        return 0, len(predicted_points), len(truth_points)
    distances = np.linalg.norm(truth_points[:, None] - predicted_points[None], axis=2)
    valid = distances <= radius
    rows, cols = linear_sum_assignment(np.where(valid, distances, 1e9))
    tp = sum(bool(valid[row, col]) for row, col in zip(rows, cols))
    return tp, len(predicted_points) - tp, len(truth_points) - tp


def operating_metrics(image_ids: list[str], truth, predictions, cutoff: float):
    tp = fp = fn = 0
    errors, positive_errors, zero_fp = [], [], []
    for image_id in image_ids:
        targets = truth.get(image_id, np.empty((0, 2), dtype=np.float32))
        points, scores = predictions.get(image_id, (np.empty((0, 2), dtype=np.float32), np.empty(0)))
        points = points[scores >= cutoff]
        itp, ifp, ifn = match_counts(targets, points)
        tp += itp; fp += ifp; fn += ifn
        error = abs(len(points) - len(targets)); errors.append(error)
        if len(targets): positive_errors.append(error)
        else: zero_fp.append(len(points))
    return {
        "point_precision": tp / (tp + fp) if tp + fp else 0.0,
        "point_recall": tp / (tp + fn) if tp + fn else 0.0,
        "point_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "fp_per_image": fp / len(image_ids),
        "count_mae_all": float(np.mean(errors)),
        "count_mae_positive": float(np.mean(positive_errors)) if positive_errors else 0.0,
        "zero_case_fp": float(np.mean(zero_fp)) if zero_fp else 0.0,
    }


def build_tables() -> list[dict]:
    rows = []
    for fold in range(3):
        for method in METHODS:
            data = json.loads(summary_path(method, fold).read_text())
            row = {
                "test_patient": PATIENTS[fold], "fold": fold, "method": method,
                "train_patient": TRAIN[fold], "validation_patient": VALIDATION[fold], "seed": 42,
                **{key: data[key] for key in SUMMARY_KEYS},
                "validation_threshold": data["operating_threshold"],
            }
            rows.append(row)
    write_csv(TABLES / "formal_per_patient_results.csv", rows)
    write_csv(TABLES / "formal_lopo_results.csv", rows)
    summary_rows = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        output = {"method": method}
        for key in SUMMARY_KEYS:
            values = [float(row[key]) for row in selected]
            output[f"{key}_mean"] = np.mean(values)
            output[f"{key}_sd"] = np.std(values, ddof=1)
        summary_rows.append(output)
    write_csv(TABLES / "formal_lopo_summary.csv", summary_rows)
    main = []
    for row in summary_rows:
        main.append({
            "method": row["method"],
            "nucleus_dice_mean_sd": f'{row["nucleus_dice_mean"]:.3f} ± {row["nucleus_dice_sd"]:.3f}',
            "cluster_dice_mean_sd": f'{row["cluster_dice_mean"]:.3f} ± {row["cluster_dice_sd"]:.3f}',
            "satellite_ap_mean_sd": f'{row["point_ap_mean"]:.3f} ± {row["point_ap_sd"]:.3f}',
            "point_f1_mean_sd": f'{row["point_f1_mean"]:.3f} ± {row["point_f1_sd"]:.3f}',
            "fp_per_image_mean_sd": f'{row["fp_per_image_mean"]:.3f} ± {row["fp_per_image_sd"]:.3f}',
            "count_mae_positive_mean_sd": f'{row["count_mae_positive_mean"]:.3f} ± {row["count_mae_positive_sd"]:.3f}',
        })
    write_csv(TABLES / "formal_main_table.csv", main)
    return rows


def build_pr_and_transfer():
    centers = load_centers()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    transfer_rows = []
    oracle_rows = []
    for fold, axis in enumerate(axes):
        image_ids = [row["image_id"] for row in read_csv(
            SOURCE / f"experiments/final/C4_w020_centerFP32_final/formal_test/fold{fold}/seed42/test_per_image_metrics.csv"
        )]
        for method, color in (("Balanced", "#2878B5"), ("Proposed C4", "#D64B3C")):
            predictions = load_candidates(candidate_path(method, fold))
            recall, precision = pr_curve({key: centers.get(key, np.empty((0, 2))) for key in image_ids}, predictions)
            ap = json.loads(summary_path(method, fold).read_text())["point_ap"]
            axis.plot(recall, precision, color=color, lw=2, label=f"{method.replace('Proposed ', '')} (AP={ap:.3f})")
            validation_threshold = threshold(method, fold)
            selected = operating_metrics(image_ids, centers, predictions, validation_threshold)
            candidates = []
            for cutoff in np.arange(0.10, 0.901, 0.05):
                candidates.append((float(cutoff), operating_metrics(image_ids, centers, predictions, float(cutoff))))
            oracle_threshold, oracle = max(candidates, key=lambda item: (item[1]["point_f1"], -item[1]["count_mae_positive"], -item[0]))
            transfer_rows.append({
                "fold": fold, "test_patient": PATIENTS[fold], "method": method,
                "validation_selected_threshold": validation_threshold,
                "test_point_f1": selected["point_f1"], "test_fp_per_image": selected["fp_per_image"],
                "test_count_mae_positive": selected["count_mae_positive"],
                "diagnostic_test_oracle_threshold": oracle_threshold,
                "diagnostic_test_oracle_f1": oracle["point_f1"],
                "f1_transfer_gap": oracle["point_f1"] - selected["point_f1"],
            })
            for cutoff, metrics in candidates:
                oracle_rows.append({"fold": fold, "test_patient": PATIENTS[fold], "method": method, "threshold": cutoff, **metrics})
        axis.set_title(f"Held-out {PATIENTS[fold]}")
        axis.set_xlabel("Recall"); axis.grid(alpha=0.25); axis.legend(fontsize=8)
    axes[0].set_ylabel("Precision")
    figure.suptitle("Satellite point-detection PR curves under LOPO-CV")
    figure.tight_layout()
    figure.savefig(FIGURES / "balanced_vs_c4_pr_curves.png", dpi=220)
    plt.close(figure)
    write_csv(TABLES / "threshold_transfer_diagnostic.csv", transfer_rows)
    write_csv(TABLES / "test_oracle_threshold_sweep_diagnostic.csv", oracle_rows)
    transfer_figure, transfer_axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for fold, axis in enumerate(transfer_axes):
        for method, color in (("Balanced", "#2878B5"), ("Proposed C4", "#D64B3C")):
            values = [row for row in oracle_rows if row["fold"] == fold and row["method"] == method]
            thresholds = np.asarray([row["threshold"] for row in values])
            f1_values = np.asarray([row["point_f1"] for row in values])
            axis.plot(thresholds, f1_values, marker="o", ms=3, color=color, lw=1.8, label=method.replace("Proposed ", ""))
            transfer = next(row for row in transfer_rows if row["fold"] == fold and row["method"] == method)
            axis.scatter([transfer["validation_selected_threshold"]], [transfer["test_point_f1"]], marker="s", s=55, color=color, edgecolor="black", zorder=4)
            axis.scatter([transfer["diagnostic_test_oracle_threshold"]], [transfer["diagnostic_test_oracle_f1"]], marker="*", s=110, color=color, edgecolor="black", zorder=5)
        axis.set_title(f"Held-out {PATIENTS[fold]}")
        axis.set_xlabel("Detection threshold"); axis.grid(alpha=0.25); axis.legend(fontsize=8)
    transfer_axes[0].set_ylabel("Test point F1")
    transfer_figure.suptitle("Validation-selected operating point (square) and diagnostic test oracle (star)")
    transfer_figure.tight_layout()
    transfer_figure.savefig(FIGURES / "threshold_transfer_f1_curves.png", dpi=220)
    plt.close(transfer_figure)


def package_release(rows: list[dict]) -> None:
    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    for name in ("C4", "Balanced", "Standard", "splits", "per_patient_results", "per_image_results", "configs", "analysis"):
        (RELEASE / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / "dataset/metadata.csv", RELEASE / "splits/metadata.csv")
    shutil.copy2(SOURCE / "dataset/splits/slide_3fold.csv", RELEASE / "splits/slide_3fold.csv")
    shutil.copy2(TABLES / "formal_per_patient_results.csv", RELEASE / "per_patient_results/formal_per_patient_results.csv")
    shutil.copy2(TABLES / "formal_main_table.csv", RELEASE / "per_patient_results/formal_main_table.csv")
    shutil.copy2(TABLES / "threshold_transfer_diagnostic.csv", RELEASE / "analysis/threshold_transfer_diagnostic.csv")
    shutil.copy2(TABLES / "test_oracle_threshold_sweep_diagnostic.csv", RELEASE / "analysis/test_oracle_threshold_sweep_diagnostic.csv")
    shutil.copy2(FIGURES / "balanced_vs_c4_pr_curves.png", RELEASE / "analysis/balanced_vs_c4_pr_curves.png")
    shutil.copy2(FIGURES / "threshold_transfer_f1_curves.png", RELEASE / "analysis/threshold_transfer_f1_curves.png")
    for fold in range(3):
        for method, source_slug, release_slug in (
            ("Standard nnU-Net", "nnunet_standard_lopo", "Standard"),
            ("Balanced", "nnunet_balanced_lopo", "Balanced"),
        ):
            base = SOURCE / f"experiments/final/{source_slug}/fold{fold}/seed42"
            destination = RELEASE / release_slug / f"fold{fold}"
            destination.mkdir(parents=True)
            for source in (
                base / "formal_test/evaluation/test_summary.json",
                base / "formal_test/evaluation/test_candidate_points.csv",
                base / "formal_test/evaluation/test_per_image_metrics.csv",
                base / "formal_test/TEST_COMPLETE_DO_NOT_RERUN",
                base / "validation_evaluation/validation_summary.json",
                base / "validation_evaluation/validation_threshold.json",
                base / "validation_evaluation/threshold_calibration.csv",
            ):
                if source.is_file(): shutil.copy2(source, destination / source.name)
            config = SOURCE / f"baseline/configs/nnunet_lopo_final/nnunet_{'standard' if method.startswith('Standard') else 'balanced'}_lopo_f{fold}_s42.yaml"
            shutil.copy2(config, RELEASE / "configs" / config.name)
        c4 = SOURCE / f"experiments/final/C4_w020_centerFP32_final/formal_test/fold{fold}/seed42"
        destination = RELEASE / "C4" / f"fold{fold}"; destination.mkdir(parents=True)
        for name in ("test_summary.json", "test_per_image_metrics.csv", "predicted_centers.csv", "TEST_COMPLETE_DO_NOT_RERUN", "config.yaml"):
            shutil.copy2(c4 / name, destination / name)
        config = SOURCE / f"baseline/configs/C4_w020_centerFP32_final/C4_w020_centerFP32_final_f{fold}_s42.yaml"
        shutil.copy2(config, RELEASE / "configs" / config.name)
        if fold == 1:
            validation_root = SOURCE / "experiments/final/C4_w020_centerFP32_final/fold1/seed42"
        else:
            validation_root = SOURCE / f"experiments/stability/C4_w020_stable_v2_gate/fold{fold}/seed42"
        for name in ("best_summary.csv",):
            source = validation_root / name
            if source.is_file(): shutil.copy2(source, destination / name)
        for name in ("metrics_summary.csv", "threshold_calibration.csv", "threshold_radius_sweep.csv"):
            source = validation_root / "validation_evaluation" / name
            if source.is_file(): shutil.copy2(source, destination / name)
        (destination / "validation_artifact_source.txt").write_text(
            str(validation_root.relative_to(SOURCE)) + "\n", encoding="utf-8"
        )
    for method, slug in (("Standard nnU-Net", "Standard"), ("Balanced", "Balanced"), ("Proposed C4", "C4")):
        for fold in range(3):
            per_image = RELEASE / slug / f"fold{fold}" / "test_per_image_metrics.csv"
            if per_image.is_file(): shutil.copy2(per_image, RELEASE / "per_image_results" / f"{slug.lower()}_fold{fold}.csv")
    checkpoints = [
        ("C4",0,"/root/autodl-tmp/cancer/experiments/stability/C4_w020_stable_v2_gate/fold0/seed42/checkpoint_best.pth",7882037,"478b69be10bfd2f66e7789b1481bceb8faa02e9538a6757ffcf14fda001d26d5",220),
        ("C4",1,"/root/autodl-tmp/cancer/experiments/final/C4_w020_centerFP32_final/fold1/seed42/checkpoint_best.pth",7881973,"2e67490d62f0780dc40963c303578a62dfcf8a6d8106352d24c9823e08bfd9ad",70),
        ("C4",2,"/root/autodl-tmp/cancer/experiments/stability/C4_w020_stable_v2_gate/fold2/seed42/checkpoint_best.pth",7882037,"04ced7a90d0fd364e6fd60c1e8a3fa9984b6156e7221b1122f9ea1f4398fa66c",140),
        ("Standard",0,"/root/autodl-tmp/cancer/nnunet/nnUNet_results/Dataset510_CCAgTLOPOF0/nnUNetTrainerSeed42__nnUNetPlans__2d/fold_0/checkpoint_best.pth",473664975,"9f12ef723d18015b4c1fda2e770dd2986d5ef9202bc12713188ffa6cbb9089ed",44),
        ("Standard",1,"/root/autodl-tmp/cancer/nnunet/nnUNet_results/Dataset511_CCAgTLOPOF1/nnUNetTrainerSeed42__nnUNetPlans__2d/fold_0/checkpoint_best.pth",473680911,"b343014764087e65db71d43b7e9bdec3d3951666256c1d03bb4037655ddf267a",79),
        ("Standard",2,"/root/autodl-tmp/cancer/nnunet/nnUNet_results/Dataset512_CCAgTLOPOF2/nnUNetTrainerSeed42__nnUNetPlans__2d/fold_0/checkpoint_best.pth",473747535,"9b7e8ea5023355b8f46abbee8db52b115e8157e8b96097c0adf945ff88fcdc99",226),
        ("Balanced",0,"/root/autodl-tmp/cancer/nnunet/nnUNet_results/Dataset510_CCAgTLOPOF0/nnUNetTrainerBalancedSamplingComposite250__nnUNetPlans__2d/fold_0/checkpoint_best.pth",473734955,"93635b96242cbb8c7367367604019fc29a447c04f595b251e78ab74d4a0bf8d4",190),
        ("Balanced",1,"/root/autodl-tmp/cancer/nnunet/nnUNet_results/Dataset511_CCAgTLOPOF1/nnUNetTrainerBalancedSamplingComposite250__nnUNetPlans__2d/fold_0/checkpoint_best.pth",473666731,"1d00ab66904673c80adbe36933355e1ae0932cc7bc5b83b56e2b2d10c54841a5",40),
        ("Balanced",2,"/root/autodl-tmp/cancer/nnunet/nnUNet_results/Dataset512_CCAgTLOPOF2/nnUNetTrainerBalancedSamplingComposite250__nnUNetPlans__2d/fold_0/checkpoint_best.pth",473689515,"763c480c3aa0c2a81d8eb020061fa686edcef939fd0a61379ea4049d237345fe",90),
    ]
    (RELEASE / "README.md").write_text(
        "# Frozen formal LOPO release\n\n"
        "This package contains the completed seed-42 three-fold leave-one-patient-out results for Standard nnU-Net, Balanced semantic nnU-Net, and C4. "
        "Checkpoint and detection-threshold selection used validation data only; every held-out patient was inferred once.\n\n"
        "`release_manifest.json` records all result-file hashes and the server path, size, epoch, and SHA-256 of every checkpoint. Large checkpoints are referenced rather than duplicated. "
        "Files under `analysis/` are descriptive diagnostics. In particular, test-oracle thresholds are post-hoc and must not replace formal validation-selected operating points.\n\n"
        "Stop rule: no retraining or parameter changes unless a demonstrated code error or data leakage invalidates this release.\n",
        encoding="utf-8",
    )
    files = {}
    for path in sorted(RELEASE.rglob("*")):
        if path.is_file(): files[str(path.relative_to(RELEASE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "release": "formal_lopo_final", "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": True, "git_commit": None,
        "git_note": "Workspace and server are not Git repositories; SHA256 file hashes are the audit anchor.",
        "stop_rule": "No retraining or parameter changes unless a demonstrated code error or data leakage invalidates the release.",
        "design": "strict 3-fold leave-one-patient/slide-out CV, seed 42",
        "test_policy": "validation-only checkpoint and threshold selection; each held-out patient inferred once",
        "checkpoint_inventory": [dict(method=m, fold=f, server_path=p, size_bytes=s, sha256=h, checkpoint_epoch=e) for m,f,p,s,h,e in checkpoints],
        "test_rows": rows, "files_sha256": files,
    }
    (RELEASE / "release_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (RELEASE / "FROZEN").write_text("formal_lopo_final\n", encoding="utf-8")


def main():
    if not SOURCE.is_dir(): raise FileNotFoundError(SOURCE)
    rows = build_tables()
    build_pr_and_transfer()
    package_release(rows)
    print(f"Wrote {len(rows)} formal rows and release {RELEASE}")


if __name__ == "__main__":
    main()
