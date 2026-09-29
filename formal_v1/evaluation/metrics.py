from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from . import CLASS_NAMES


def confusion_matrix(target: np.ndarray, prediction: np.ndarray, classes: int = 4):
    valid = (target >= 0) & (target < classes)
    encoded = classes * target[valid].astype(np.int64) + prediction[valid]
    return np.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def pixel_rows(matrix: np.ndarray, image_id: str | None = None) -> list[dict]:
    rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        tp = int(matrix[class_id, class_id])
        fn = int(matrix[class_id, :].sum() - tp)
        fp = int(matrix[:, class_id].sum() - tp)
        dice_den = 2 * tp + fp + fn
        iou_den = tp + fp + fn
        row = {
            "class_id": class_id,
            "class_name": class_name,
            "dice": 2 * tp / dice_den if dice_den else float("nan"),
            "iou": tp / iou_den if iou_den else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "tp_pixels": tp,
            "fp_pixels": fp,
            "fn_pixels": fn,
        }
        if image_id is not None:
            row = {"image_id": image_id, **row}
        rows.append(row)
    return rows


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    true_count: int
    predicted_count: int


def _components(mask: np.ndarray):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return labels.astype(np.int32), int(count)


def _iou_matrix(true_labels: np.ndarray, n_true: int, pred_labels: np.ndarray, n_pred: int):
    if not n_true or not n_pred:
        return np.zeros((n_true, n_pred), dtype=np.float64)
    joint = np.bincount(
        (true_labels.ravel() * (n_pred + 1) + pred_labels.ravel()).astype(np.int64),
        minlength=(n_true + 1) * (n_pred + 1),
    ).reshape(n_true + 1, n_pred + 1)
    intersection = joint[1:, 1:].astype(np.float64)
    true_area = np.bincount(true_labels.ravel(), minlength=n_true + 1)[1:, None]
    pred_area = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)[None, 1:]
    return intersection / np.maximum(true_area + pred_area - intersection, 1)


def match_objects(
    target: np.ndarray,
    prediction: np.ndarray,
    method: str,
    threshold: float,
) -> MatchResult:
    true_labels, n_true = _components(target)
    pred_labels, n_pred = _components(prediction)
    if not n_true or not n_pred:
        return MatchResult(0, n_pred, n_true, n_true, n_pred)

    if method == "iou":
        scores = _iou_matrix(true_labels, n_true, pred_labels, n_pred)
        valid = scores >= threshold
    elif method == "distance":
        true_centres = np.asarray(ndimage.center_of_mass(target, true_labels, range(1, n_true + 1)))
        pred_centres = np.asarray(ndimage.center_of_mass(prediction, pred_labels, range(1, n_pred + 1)))
        distances = np.linalg.norm(true_centres[:, None, :] - pred_centres[None, :, :], axis=2)
        valid = distances <= threshold
        scores = np.maximum(0.0, 1.0 - distances / max(threshold, 1e-12))
    else:
        raise ValueError(f"Unknown object matching method: {method}")

    costs = np.where(valid, -scores, 1e6)
    rows, columns = linear_sum_assignment(costs)
    tp = sum(bool(valid[row, column]) for row, column in zip(rows, columns))
    return MatchResult(tp, n_pred - tp, n_true - tp, n_true, n_pred)


def satellite_count_mae_per_gt_nucleus(target: np.ndarray, prediction: np.ndarray) -> tuple[float, int]:
    # This is intentionally GT-nucleus-conditioned: it measures satellite count
    # error independently of nucleus segmentation error. Global object metrics
    # still penalize predictions outside every GT nucleus as false positives.
    nuclear_roi = np.isin(target, (1, 2, 3))
    nucleus_labels, n_nuclei = _components(nuclear_roi)
    if not n_nuclei:
        return float("nan"), 0

    true_sat, n_true = _components(target == 3)
    pred_sat, n_pred = _components(prediction == 3)
    true_counts = np.zeros(n_nuclei + 1, dtype=np.int64)
    pred_counts = np.zeros(n_nuclei + 1, dtype=np.int64)
    for labels, count, output in (
        (true_sat, n_true, true_counts),
        (pred_sat, n_pred, pred_counts),
    ):
        if not count:
            continue
        centres = ndimage.center_of_mass(labels > 0, labels, range(1, count + 1))
        for row, column in centres:
            r = int(np.clip(round(row), 0, target.shape[0] - 1))
            c = int(np.clip(round(column), 0, target.shape[1] - 1))
            nucleus_id = nucleus_labels[r, c]
            if nucleus_id:
                output[nucleus_id] += 1
    return float(np.abs(true_counts[1:] - pred_counts[1:]).mean()), n_nuclei


def object_count_case_metrics(rows: list[dict]) -> dict:
    """Summarize count and detection metrics without zero-case dilution.

    ``rows`` must contain one row per image for a single object class.  An empty
    prediction is a valid prediction and therefore always contributes to MAE.
    """
    if not rows:
        return {
            "count_mae_all": float("nan"),
            "count_mae_positive": float("nan"),
            "zero_case_fp": float("nan"),
            "positive_case_recall": float("nan"),
            "positive_image_count": 0,
            "zero_image_count": 0,
        }

    positive = [row for row in rows if int(row["true_count"]) > 0]
    zero = [row for row in rows if int(row["true_count"]) == 0]
    positive_tp = sum(int(row["tp"]) for row in positive)
    positive_fn = sum(int(row["fn"]) for row in positive)
    return {
        "count_mae_all": sum(float(row["absolute_count_error"]) for row in rows) / len(rows),
        "count_mae_positive": (
            sum(float(row["absolute_count_error"]) for row in positive) / len(positive)
            if positive
            else float("nan")
        ),
        "zero_case_fp": (
            sum(int(row["predicted_count"]) for row in zero) / len(zero)
            if zero
            else float("nan")
        ),
        "positive_case_recall": (
            positive_tp / (positive_tp + positive_fn)
            if positive_tp + positive_fn
            else float("nan")
        ),
        "positive_image_count": len(positive),
        "zero_image_count": len(zero),
    }


def component_split_merge_counts(target: np.ndarray, prediction: np.ndarray) -> tuple[int, int]:
    """Return (fragmented GT objects, merged predicted objects) by overlap."""
    true_labels, n_true = _components(target)
    pred_labels, n_pred = _components(prediction)
    if not n_true or not n_pred:
        return 0, 0
    overlap = np.bincount(
        (true_labels.ravel() * (n_pred + 1) + pred_labels.ravel()).astype(np.int64),
        minlength=(n_true + 1) * (n_pred + 1),
    ).reshape(n_true + 1, n_pred + 1)[1:, 1:] > 0
    fragmented = int(np.count_nonzero(overlap.sum(axis=1) > 1))
    merged = int(np.count_nonzero(overlap.sum(axis=0) > 1))
    return fragmented, merged
