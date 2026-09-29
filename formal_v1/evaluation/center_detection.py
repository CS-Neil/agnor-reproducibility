from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


def components_to_centers(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if not count:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(
        ndimage.center_of_mass(mask, labels, range(1, count + 1)), dtype=np.float32
    )


def draw_gaussians(
    shape: tuple[int, int], centers: np.ndarray, sigma: float = 2.5, truncate: float = 3.0
) -> np.ndarray:
    output = np.zeros(shape, dtype=np.float32)
    radius = int(math.ceil(sigma * truncate))
    for row, column in np.asarray(centers, dtype=np.float32).reshape(-1, 2):
        peak_row = int(np.clip(round(float(row)), 0, shape[0] - 1))
        peak_column = int(np.clip(round(float(column)), 0, shape[1] - 1))
        row0 = max(0, peak_row - radius)
        row1 = min(shape[0], peak_row + radius + 1)
        col0 = max(0, peak_column - radius)
        col1 = min(shape[1], peak_column + radius + 1)
        yy, xx = np.ogrid[row0:row1, col0:col1]
        values = np.exp(
            -((yy - peak_row) ** 2 + (xx - peak_column) ** 2) / (2 * sigma**2)
        )
        patch = output[row0:row1, col0:col1]
        np.maximum(patch, values.astype(np.float32), out=patch)
    return output


def modified_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Numerically stable FP32 CenterNet focal loss for Gaussian heatmaps."""
    logits = logits.float()
    if target.ndim == logits.ndim - 1:
        target = target[:, None]
    target = target.float()
    probability = logits.sigmoid()
    positive = target >= (1 - 1e-4)
    negative = ~positive
    negative_weight = (1 - target).pow(4)
    positive_loss = (
        F.logsigmoid(logits) * (1 - probability).pow(2) * positive
    ).sum()
    negative_loss = (
        F.logsigmoid(-logits)
        * probability.pow(2)
        * negative_weight
        * negative
    ).sum()
    positive_count = positive.sum().to(probability.dtype)
    if positive_count.item() == 0:
        return -negative_loss
    return -(positive_loss + negative_loss) / positive_count


def extract_local_maxima(
    heatmap: np.ndarray, threshold: float, nms_radius: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got {heatmap.shape}")
    maximum = ndimage.maximum_filter(
        heatmap, size=2 * nms_radius + 1, mode="constant", cval=-np.inf
    )
    candidates = (heatmap >= threshold) & (heatmap == maximum)
    plateau_labels, count = ndimage.label(candidates, structure=np.ones((3, 3), dtype=np.uint8))
    points: list[tuple[float, float]] = []
    scores: list[float] = []
    for label_id in range(1, count + 1):
        rows, columns = np.nonzero(plateau_labels == label_id)
        values = heatmap[rows, columns]
        index = int(np.argmax(values))
        points.append((float(rows[index]), float(columns[index])))
        scores.append(float(values[index]))
    if not points:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    order = np.argsort(np.asarray(scores))[::-1]
    return np.asarray(points, dtype=np.float32)[order], np.asarray(scores, dtype=np.float32)[order]


@dataclass(frozen=True)
class PointMatch:
    tp: int
    fp: int
    fn: int
    localization_errors: tuple[float, ...]


def match_points(
    true_points: np.ndarray, predicted_points: np.ndarray, max_distance: float
) -> PointMatch:
    true_points = np.asarray(true_points, dtype=np.float32).reshape(-1, 2)
    predicted_points = np.asarray(predicted_points, dtype=np.float32).reshape(-1, 2)
    if not len(true_points) or not len(predicted_points):
        return PointMatch(0, len(predicted_points), len(true_points), ())
    distances = np.linalg.norm(
        true_points[:, None, :] - predicted_points[None, :, :], axis=2
    )
    valid = distances <= max_distance
    rows, columns = linear_sum_assignment(np.where(valid, distances, 1e9))
    errors = tuple(
        float(distances[row, column])
        for row, column in zip(rows, columns)
        if valid[row, column]
    )
    return PointMatch(
        len(errors), len(predicted_points) - len(errors), len(true_points) - len(errors), errors
    )


def point_average_precision(
    true_points_by_image: dict[str, np.ndarray],
    predictions_by_image: dict[str, tuple[np.ndarray, np.ndarray]],
    max_distance: float,
) -> float:
    """Dataset-level all-points AP using score-ordered one-to-one distance matching."""
    total_true = sum(
        len(np.asarray(points).reshape(-1, 2))
        for points in true_points_by_image.values()
    )
    if total_true == 0:
        return 0.0

    detections: list[tuple[float, str, int, np.ndarray]] = []
    for image_id, (points, scores) in predictions_by_image.items():
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(points) != len(scores):
            raise ValueError(f"Point/score count mismatch for {image_id}")
        detections.extend(
            (float(score), image_id, index, point)
            for index, (point, score) in enumerate(zip(points, scores))
        )
    if not detections:
        return 0.0
    detections.sort(key=lambda value: (-value[0], value[1], value[2]))

    matched_truth = {
        image_id: np.zeros(len(np.asarray(points).reshape(-1, 2)), dtype=bool)
        for image_id, points in true_points_by_image.items()
    }
    true_flags: list[float] = []
    false_flags: list[float] = []
    for _, image_id, _, predicted_point in detections:
        truths = np.asarray(
            true_points_by_image.get(image_id, np.empty((0, 2), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1, 2)
        used = matched_truth.setdefault(image_id, np.zeros(len(truths), dtype=bool))
        if len(truths):
            distances = np.linalg.norm(truths - predicted_point[None], axis=1)
            distances[used] = np.inf
            truth_index = int(np.argmin(distances))
            is_true = bool(distances[truth_index] <= max_distance)
        else:
            truth_index = -1
            is_true = False
        if is_true:
            used[truth_index] = True
        true_flags.append(float(is_true))
        false_flags.append(float(not is_true))

    true_cumulative = np.cumsum(true_flags)
    false_cumulative = np.cumsum(false_flags)
    recall = true_cumulative / total_true
    precision = true_cumulative / np.maximum(true_cumulative + false_cumulative, 1)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changing = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[changing + 1] - recall[changing]) * precision[changing + 1]))


def points_to_mask(
    shape: tuple[int, int], points: np.ndarray, radius: int
) -> np.ndarray:
    output = np.zeros(shape, dtype=bool)
    for row, column in np.asarray(points).reshape(-1, 2):
        row0 = max(0, int(round(row)) - radius)
        row1 = min(shape[0], int(round(row)) + radius + 1)
        col0 = max(0, int(round(column)) - radius)
        col1 = min(shape[1], int(round(column)) + radius + 1)
        yy, xx = np.ogrid[row0:row1, col0:col1]
        output[row0:row1, col0:col1] |= (
            (yy - row) ** 2 + (xx - column) ** 2 <= radius**2
        )
    return output
