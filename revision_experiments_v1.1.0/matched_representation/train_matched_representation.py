#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ccagt_baseline.center_detection import (
    components_to_centers,
    draw_gaussians,
    extract_local_maxima,
    match_points,
    modified_focal_loss,
    point_average_precision,
    points_to_mask,
)
from ccagt_baseline.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_pair,
    resolve_records,
    sliding_positions,
    write_resolved_split,
)
from ccagt_baseline.losses import dice_ce_loss
from ccagt_baseline.metrics import confusion_matrix, pixel_rows
from ccagt_baseline.utils import append_row, load_config, save_yaml, seed_everything


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_centers(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[list[float]]] = defaultdict(list)
    for row in read_csv(path):
        values[row["image_id"]].append([float(row["row"]), float(row["column"])])
    return {
        image_id: np.asarray(points, dtype=np.float32).reshape(-1, 2)
        for image_id, points in values.items()
    }


def normalize_image(image: np.ndarray) -> torch.Tensor:
    array = image.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


class BalancedCenterPatchDataset(Dataset):
    categories = ("satellite", "cluster", "nucleus", "random")

    def __init__(
        self,
        records,
        centers_by_id: dict[str, np.ndarray],
        summaries: list[dict[str, str]],
        patch_size: tuple[int, int],
        samples_per_epoch: int,
        probabilities: tuple[float, float, float, float],
        sigma: float,
        truncate: float,
        mode: str,
        augment: bool,
    ):
        self.records = {record.image_id: record for record in records}
        self.centers_by_id = centers_by_id
        self.patch_width, self.patch_height = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.probabilities /= self.probabilities.sum()
        self.sigma = sigma
        self.truncate = truncate
        self.mode = mode
        self.augment = augment
        allowed = set(self.records)
        summaries = [row for row in summaries if row["image_id"] in allowed]
        self.pools = {
            "satellite": [row["image_id"] for row in summaries if int(row["satellite_pixels"]) > 0],
            "cluster": [row["image_id"] for row in summaries if int(row["cluster_pixels"]) > 0],
            "nucleus": [row["image_id"] for row in summaries if int(row["nucleus_pixels"]) > 0],
            "random": [row["image_id"] for row in summaries],
        }
        if any(not pool for pool in self.pools.values()):
            raise RuntimeError(f"Empty balanced sampling pool: {self.pools}")
        print(
            "Balanced pools:",
            {key: len(value) for key, value in self.pools.items()},
            "probabilities:",
            dict(zip(self.categories, self.probabilities.tolist())),
            flush=True,
        )

    def __len__(self):
        return self.samples_per_epoch

    @staticmethod
    def _crop_start(coordinate: int, patch: int, length: int, rng) -> int:
        if patch >= length:
            return 0
        low = max(0, coordinate - patch + 1)
        high = min(coordinate, length - patch)
        return int(rng.integers(low, high + 1))

    def __getitem__(self, _):
        rng = np.random.default_rng(
            np.random.randint(0, np.iinfo(np.int32).max, dtype=np.int64)
        )
        category = str(rng.choice(self.categories, p=self.probabilities))
        image_id = str(rng.choice(self.pools[category]))
        image, mask = load_pair(self.records[image_id])
        height, width = mask.shape
        if category == "random":
            crop_x = int(rng.integers(0, max(1, width - self.patch_width + 1)))
            crop_y = int(rng.integers(0, max(1, height - self.patch_height + 1)))
        else:
            class_id = {"nucleus": 1, "cluster": 2, "satellite": 3}[category]
            positions = np.argwhere(mask == class_id)
            if not len(positions):
                raise RuntimeError(f"Sampling pool inconsistency: {category} {image_id}")
            row, column = positions[int(rng.integers(0, len(positions)))]
            crop_x = self._crop_start(int(column), self.patch_width, width, rng)
            crop_y = self._crop_start(int(row), self.patch_height, height, rng)
        image = image[crop_y : crop_y + self.patch_height, crop_x : crop_x + self.patch_width]
        mask = mask[crop_y : crop_y + self.patch_height, crop_x : crop_x + self.patch_width]
        instance_counts = {
            name: int(ndimage.label(mask == class_id, structure=np.ones((3, 3), dtype=np.uint8))[1])
            for name, class_id in (("nucleus", 1), ("cluster", 2), ("satellite", 3))
        }
        centers = self.centers_by_id.get(image_id, np.empty((0, 2), dtype=np.float32)).copy()
        centers[:, 0] -= crop_y
        centers[:, 1] -= crop_x
        inside = (
            (centers[:, 0] >= 0)
            & (centers[:, 0] < mask.shape[0])
            & (centers[:, 1] >= 0)
            & (centers[:, 1] < mask.shape[1])
        )
        centers = centers[inside]

        if self.augment and rng.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
            centers[:, 1] = mask.shape[1] - 1 - centers[:, 1]
        if self.augment and rng.random() < 0.5:
            image, mask = image[::-1], mask[::-1]
            centers[:, 0] = mask.shape[0] - 1 - centers[:, 0]
        image = image.astype(np.float32) / 255.0
        if self.augment:
            gamma = float(rng.uniform(0.85, 1.15))
            scale = float(rng.uniform(0.90, 1.10))
            shift = float(rng.uniform(-0.04, 0.04))
            image = np.clip(image, 0, 1) ** gamma
            image = np.clip(image * scale + shift, 0, 1)
            if rng.random() < 0.15:
                image = np.clip(image + rng.normal(0, 0.01, image.shape), 0, 1)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1)).astype(np.float32)
        )
        segmentation = np.ascontiguousarray(mask).astype(np.int64)
        if self.mode in {
            "c2_center_only",
            "c3_decoupled_mask_aux",
            "c4_decoupled_no_mask",
        }:
            segmentation[segmentation == 3] = 1
        heatmap = draw_gaussians(mask.shape, centers, self.sigma, self.truncate)
        return {
            "image": image_tensor,
            "segmentation": torch.from_numpy(segmentation),
            "satellite_mask": torch.from_numpy(
                np.ascontiguousarray(mask == 3).astype(np.float32)[None]
            ),
            "heatmap": torch.from_numpy(heatmap[None]),
            "sampling_category": category,
            "center_count": len(centers),
            "image_id": image_id,
            "crop_x": crop_x,
            "crop_y": crop_y,
            "nucleus_instance_count": instance_counts["nucleus"],
            "cluster_instance_count": instance_counts["cluster"],
            "satellite_instance_count": instance_counts["satellite"],
        }


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed + worker_id)


class ConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.InstanceNorm2d(outputs, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.InstanceNorm2d(outputs, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.block(inputs)


class CenterMultiTaskUNet(nn.Module):
    def __init__(
        self, semantic_classes: int, base: int = 16, mask_auxiliary: bool = False
    ):
        super().__init__()
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.bottom = ConvBlock(base * 8, base * 16)
        self.dec4 = ConvBlock(base * 16 + base * 8, base * 8)
        self.dec3 = ConvBlock(base * 8 + base * 4, base * 4)
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2 + base, base)
        self.semantic_head = nn.Conv2d(base, semantic_classes, 1)
        self.center_head = nn.Conv2d(base, 1, 1)
        self.mask_auxiliary_head = nn.Conv2d(base, 1, 1) if mask_auxiliary else None
        nn.init.zeros_(self.center_head.weight)
        nn.init.constant_(self.center_head.bias, -4.595)
        if self.mask_auxiliary_head is not None:
            nn.init.zeros_(self.mask_auxiliary_head.weight)
            nn.init.constant_(self.mask_auxiliary_head.bias, -4.595)

    @staticmethod
    def upsample(value, reference):
        return F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, inputs, center_branch_fp32: bool = False):
        e1 = self.enc1(inputs)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        bottom = self.bottom(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat((self.upsample(bottom, e4), e4), 1))
        d3 = self.dec3(torch.cat((self.upsample(d4, e3), e3), 1))
        d2 = self.dec2(torch.cat((self.upsample(d3, e2), e2), 1))
        d1 = self.dec1(torch.cat((self.upsample(d2, e1), e1), 1))
        mask_logits = (
            self.mask_auxiliary_head(d1)
            if self.mask_auxiliary_head is not None
            else None
        )
        if center_branch_fp32:
            with torch.autocast(device_type=d1.device.type, enabled=False):
                center_logits = self.center_head(d1.float())
        else:
            center_logits = self.center_head(d1)
        return self.semantic_head(d1), center_logits, mask_logits


def binary_dice_bce_loss(logits, target, epsilon: float = 1e-6):
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(0, 2, 3))
    denominator = probability.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = 1 - ((2 * intersection + epsilon) / (denominator + epsilon)).mean()
    return F.binary_cross_entropy_with_logits(logits, target) + dice


def assert_finite_gradients(model: nn.Module) -> None:
    gradients = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    flags = [torch.isfinite(gradient).all() for _, gradient in gradients]
    if flags and not bool(torch.stack(flags).all().item()):
        for name, gradient in gradients:
            if not bool(torch.isfinite(gradient).all().item()):
                raise FloatingPointError(f"Non-finite gradient in {name}")


def run_training_epoch(
    model,
    loader,
    device,
    optimizer,
    scaler,
    center_weight,
    mask_auxiliary_weight,
    epoch,
    satellite_target,
):
    model.train()
    totals = defaultdict(float)
    samples = 0
    iterator = tqdm(loader, leave=False, desc="center-train")
    for batch_index, batch in enumerate(iterator, 1):
        images = batch["image"].to(device, non_blocking=True)
        segmentations = batch["segmentation"].to(device, non_blocking=True)
        satellite_masks = batch["satellite_mask"].to(device, non_blocking=True)
        heatmaps = batch["heatmap"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            semantic_logits, center_logits, mask_logits = model(
                images, center_branch_fp32=True
            )
            segmentation_loss = dice_ce_loss(
                semantic_logits, segmentations, include_background=False
            )
            mask_auxiliary_loss = (
                binary_dice_bce_loss(mask_logits, satellite_masks)
                if mask_logits is not None
                else segmentation_loss.new_zeros(())
            )
        with torch.autocast(device_type=device.type, enabled=False):
            if satellite_target == "mask":
                center_loss = binary_dice_bce_loss(
                    center_logits.float(), satellite_masks.float()
                )
            else:
                center_loss = modified_focal_loss(
                    center_logits.float(), heatmaps.float()
                )
        if not bool(torch.isfinite(center_loss).item()):
            raise FloatingPointError(
                f"Non-finite center loss at epoch {epoch}, batch {batch_index}: "
                f"{center_loss.item()}"
            )
        loss = (
            segmentation_loss.float()
            + center_weight * center_loss
            + mask_auxiliary_weight * mask_auxiliary_loss.float()
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"Non-finite total loss at epoch {epoch}, batch {batch_index}: "
                f"{loss.item()}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert_finite_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        samples += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        totals["segmentation_loss"] += float(segmentation_loss.detach()) * batch_size
        totals["center_loss"] += float(center_loss.detach()) * batch_size
        totals["mask_auxiliary_loss"] += float(mask_auxiliary_loss.detach()) * batch_size
        totals["center_count"] += int(batch["center_count"].sum())
        for category in batch["sampling_category"]:
            totals[f"sample_{category}"] += 1
        totals["finite_gradient_checks"] += 1
    return {
        "train_loss": totals["loss"] / max(samples, 1),
        "segmentation_loss": totals["segmentation_loss"] / max(samples, 1),
        "center_loss": totals["center_loss"] / max(samples, 1),
        "mask_auxiliary_loss": totals["mask_auxiliary_loss"] / max(samples, 1),
        "finite_gradient_checks": int(totals["finite_gradient_checks"]),
        "amp_grad_scale": float(scaler.get_scale()),
        "sampled_centers": int(totals["center_count"]),
        **{
            f"sample_{category}": int(totals[f"sample_{category}"])
            for category in BalancedCenterPatchDataset.categories
        },
    }


def importance_map(height: int, width: int) -> np.ndarray:
    vertical = np.hanning(height).astype(np.float32)
    horizontal = np.hanning(width).astype(np.float32)
    output = vertical[:, None] * horizontal[None, :]
    return np.maximum(output, 0.05)


@torch.no_grad()
def infer_image(model, image, patch_size, overlap, device, semantic_classes):
    model.eval()
    patch_width, patch_height = patch_size
    height, width = image.shape[:2]
    y_positions = sliding_positions(height, patch_height, overlap)
    x_positions = sliding_positions(width, patch_width, overlap)
    weights = importance_map(patch_height, patch_width)
    semantic_sum = np.zeros((semantic_classes, height, width), dtype=np.float32)
    center_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    for y in y_positions:
        for x in x_positions:
            patch = image[y : y + patch_height, x : x + patch_width]
            inputs = normalize_image(patch)[None].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                semantic_logits, center_logits, _ = model(
                    inputs, center_branch_fp32=True
                )
            semantic = semantic_logits.softmax(1)[0].float().cpu().numpy()
            center = center_logits.sigmoid()[0, 0].float().cpu().numpy()
            semantic_sum[:, y : y + patch_height, x : x + patch_width] += semantic * weights
            center_sum[y : y + patch_height, x : x + patch_width] += center * weights
            weight_sum[y : y + patch_height, x : x + patch_width] += weights
    return semantic_sum / weight_sum[None], center_sum / weight_sum


def summarize_point_rows(rows: list[dict], localization_errors: list[float]) -> dict:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    positive = [row for row in rows if int(row["true_count"]) > 0]
    zero = [row for row in rows if int(row["true_count"]) == 0]
    return {
        "point_precision": tp / (tp + fp) if tp + fp else 0.0,
        "point_recall": tp / (tp + fn) if tp + fn else 0.0,
        "point_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "point_tp": tp,
        "point_fp": fp,
        "point_fn": fn,
        "predicted_center_count": sum(int(row["predicted_count"]) for row in rows),
        "true_center_count": sum(int(row["true_count"]) for row in rows),
        "fp_per_image": fp / max(len(rows), 1),
        "count_mae_all": np.mean([row["absolute_count_error"] for row in rows]),
        "count_mae_positive": (
            np.mean([row["absolute_count_error"] for row in positive]) if positive else float("nan")
        ),
        "zero_case_fp": (
            np.mean([row["predicted_count"] for row in zero]) if zero else float("nan")
        ),
        "positive_case_recall": (
            sum(int(row["tp"]) for row in positive)
            / max(sum(int(row["tp"]) + int(row["fn"]) for row in positive), 1)
        ),
        "localization_error": (
            float(np.mean(localization_errors)) if localization_errors else float("nan")
        ),
    }


def checkpoint_protocol(config) -> str:
    return str(config["checkpoint_selection"].get("protocol", "protocol-v1"))


def checkpoint_score(metrics: dict, config) -> float:
    protocol = checkpoint_protocol(config)
    if protocol == "protocol-v2":
        return float(
            0.20 * metrics["nucleus_dice"]
            + 0.20 * metrics["cluster_dice"]
            + 0.60 * metrics["point_ap"]
        )
    if protocol == "protocol-v1":
        return float(
            0.25 * metrics["nucleus_dice"]
            + 0.25 * metrics["cluster_dice"]
            + 0.50 * metrics["point_f1"]
        )
    raise ValueError(f"Unknown checkpoint selection protocol: {protocol}")


@torch.no_grad()
def evaluate_full_images(
    model,
    records,
    centers_by_id,
    config,
    device,
    output: Path | None = None,
    retain_heatmaps: bool = False,
    threshold_override: float | None = None,
):
    mode = config["experiment"]["mode"]
    semantic_classes = 4 if mode == "c1_auxiliary" else 3
    patch_size = tuple(int(value) for value in config["training"]["patch_size"])
    overlap = float(config["inference"]["overlap"])
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else float(config["point_evaluation"]["checkpoint_threshold"])
    )
    nms_radius = int(config["point_evaluation"]["nms_radius_px"])
    match_radius = float(config["point_evaluation"]["checkpoint_match_radius_px"])
    render_radius = int(config["point_evaluation"]["render_radius_px"])
    matrix = np.zeros((4, 4), dtype=np.int64)
    semantic_matrix = np.zeros(
        (4, 4) if mode == "c1_auxiliary" else (3, 3), dtype=np.int64
    )
    point_rows: list[dict] = []
    localization_errors: list[float] = []
    ap_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ap_truth: dict[str, np.ndarray] = {}
    heatmap_cache: dict[str, np.ndarray] = {}
    point_output: list[dict] = []
    if output is not None:
        (output / "semantic_predictions").mkdir(parents=True, exist_ok=True)
        (output / "center_heatmaps").mkdir(parents=True, exist_ok=True)
    for record in tqdm(records, desc="full-validation", leave=False):
        image, target = load_pair(record)
        semantic_probability, center_heatmap = infer_image(
            model, image, patch_size, overlap, device, semantic_classes
        )
        predicted_points, predicted_scores = extract_local_maxima(
            center_heatmap, threshold, nms_radius
        )
        true_points = centers_by_id.get(record.image_id, np.empty((0, 2), dtype=np.float32))
        ap_points, ap_scores = extract_local_maxima(
            center_heatmap,
            float(config["point_evaluation"].get("ap_candidate_threshold", 0.05)),
            nms_radius,
        )
        ap_predictions[record.image_id] = (ap_points, ap_scores)
        ap_truth[record.image_id] = true_points
        matched = match_points(true_points, predicted_points, match_radius)
        localization_errors.extend(matched.localization_errors)
        point_rows.append(
            {
                "image_id": record.image_id,
                "tp": matched.tp,
                "fp": matched.fp,
                "fn": matched.fn,
                "true_count": len(true_points),
                "predicted_count": len(predicted_points),
                "absolute_count_error": abs(len(true_points) - len(predicted_points)),
            }
        )
        semantic_prediction = semantic_probability.argmax(0).astype(np.uint8)
        if mode == "c1_auxiliary":
            semantic_matrix += confusion_matrix(target, semantic_prediction)
        else:
            semantic_target = target.copy()
            semantic_target[semantic_target == 3] = 1
            semantic_matrix += confusion_matrix(
                semantic_target, semantic_prediction, classes=3
            )
        prediction = semantic_prediction.copy()
        if mode != "c1_auxiliary":
            prediction[points_to_mask(target.shape, predicted_points, render_radius)] = 3
        matrix += confusion_matrix(target, prediction)
        for point_index, ((row, column), score) in enumerate(
            zip(predicted_points, predicted_scores), 1
        ):
            point_output.append(
                {
                    "image_id": record.image_id,
                    "point_id": point_index,
                    "row": row,
                    "column": column,
                    "score": score,
                }
            )
        if retain_heatmaps:
            heatmap_cache[record.image_id] = center_heatmap.astype(np.float16)
        if output is not None:
            Image.fromarray(prediction).save(
                output / "semantic_predictions" / f"{record.image_id}.png"
            )
            Image.fromarray(np.clip(center_heatmap * 255, 0, 255).astype(np.uint8)).save(
                output / "center_heatmaps" / f"{record.image_id}.png"
            )
    pixels = pixel_rows(matrix)
    def class_dice(matrix_value, class_id):
        true_positive = int(matrix_value[class_id, class_id])
        false_positive = int(matrix_value[:, class_id].sum() - true_positive)
        false_negative = int(matrix_value[class_id].sum() - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        return 2 * true_positive / denominator if denominator else 0.0

    metrics = {
        "nucleus_dice": class_dice(semantic_matrix, 1),
        "cluster_dice": class_dice(semantic_matrix, 2),
        "satellite_pixel_dice": pixels[3]["dice"],
        "point_ap": point_average_precision(ap_truth, ap_predictions, match_radius),
        **summarize_point_rows(point_rows, localization_errors),
        "validation_images": len(records),
        "operating_threshold": threshold,
        "match_radius_px": match_radius,
    }
    metrics["selection_protocol"] = checkpoint_protocol(config)
    metrics["composite_score"] = checkpoint_score(metrics, config)
    if output is not None:
        write_rows(output / "metrics_points_per_image.csv", point_rows)
        write_rows(output / "predicted_points.csv", point_output)
        write_rows(output / "metrics_pixel.csv", pixels)
        write_rows(output / "metrics_summary.csv", [metrics])
    return metrics, heatmap_cache


def threshold_radius_sweep(
    records,
    centers_by_id,
    heatmap_cache,
    config,
    match_radii: tuple[float, ...] = (3.0, 5.0, 7.0),
):
    rows: list[dict] = []
    nms_radius = int(config["point_evaluation"]["nms_radius_px"])
    start, stop, step = (
        float(value) for value in config["point_evaluation"]["threshold_sweep"]
    )
    thresholds = np.arange(start, stop + step / 2, step)
    peak_cache = {
        record.image_id: extract_local_maxima(
            heatmap_cache[record.image_id].astype(np.float32),
            float(thresholds.min()),
            nms_radius,
        )
        for record in records
    }
    for match_radius in match_radii:
        for threshold in thresholds:
            image_rows = []
            localization_errors: list[float] = []
            for record in records:
                all_points, all_scores = peak_cache[record.image_id]
                points = all_points[all_scores >= threshold]
                truth = centers_by_id.get(
                    record.image_id, np.empty((0, 2), dtype=np.float32)
                )
                matched = match_points(truth, points, match_radius)
                localization_errors.extend(matched.localization_errors)
                image_rows.append(
                    {
                        "tp": matched.tp,
                        "fp": matched.fp,
                        "fn": matched.fn,
                        "true_count": len(truth),
                        "predicted_count": len(points),
                        "absolute_count_error": abs(len(truth) - len(points)),
                    }
                )
            rows.append(
                {
                    "threshold": round(float(threshold), 2),
                    "match_radius_px": match_radius,
                    **summarize_point_rows(image_rows, localization_errors),
                }
            )
    rows.sort(key=lambda row: row["point_f1"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    satellite_target = str(config["experiment"]["satellite_target"])
    if satellite_target not in {"mask", "center"}:
        raise ValueError(f"Unknown satellite target: {satellite_target}")
    if config["experiment"]["mode"] != "c4_decoupled_no_mask":
        raise ValueError("Matched ablation requires the same three-class regional head")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output / "config.yaml")
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda")

    all_records = resolve_records(config)
    write_resolved_split(all_records, output / "resolved_split.csv")
    train_records = [record for record in all_records if record.role == "train"]
    validation_records = [record for record in all_records if record.role == "validation"]
    label_root = Path(config["center_labels"]["root"])
    centers_by_id = load_centers(label_root / "centers.csv")
    summaries = read_csv(label_root / "image_summary.csv")
    batch_size = int(config["training"]["batch_size"])
    dataset = BalancedCenterPatchDataset(
        train_records,
        centers_by_id,
        summaries,
        tuple(int(value) for value in config["training"]["patch_size"]),
        int(config["training"]["iterations_per_epoch"]) * batch_size,
        tuple(float(value) for value in config["sampling"]["probabilities"]),
        float(config["center_labels"]["sigma_px"]),
        float(config["center_labels"]["truncate_sigma"]),
        config["experiment"]["mode"],
        augment=True,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=generator,
        prefetch_factor=2,
    )
    mode = config["experiment"]["mode"]
    semantic_classes = 4 if mode == "c1_auxiliary" else 3
    model = CenterMultiTaskUNet(
        semantic_classes,
        base=int(config["model"]["base_channels"]),
        mask_auxiliary=mode == "c3_decoupled_mask_aux",
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        momentum=0.99,
        nesterov=True,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=float(config["training"].get("grad_scaler_init_scale", 65536.0)),
        growth_interval=int(
            config["training"].get("grad_scaler_growth_interval", 2000)
        ),
    )
    max_epochs = int(config["training"]["max_epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: max(0.0, 1 - epoch / max_epochs) ** 0.9
    )
    interval = int(config["checkpoint_selection"]["interval_epochs"])
    patience_checks = int(config["training"]["early_stopping_patience_epochs"]) // interval
    center_weight = float(config["training"]["center_loss_weight"])
    mask_auxiliary_weight = float(config["training"].get("mask_auxiliary_weight", 0.0))
    best_score = -float("inf")
    bad_checks = 0
    started = time.time()
    validation_fields = (
        "nucleus_dice",
        "cluster_dice",
        "satellite_pixel_dice",
        "point_ap",
        "point_precision",
        "point_recall",
        "point_f1",
        "point_tp",
        "point_fp",
        "point_fn",
        "predicted_center_count",
        "true_center_count",
        "fp_per_image",
        "count_mae_all",
        "count_mae_positive",
        "zero_case_fp",
        "positive_case_recall",
        "localization_error",
        "validation_images",
        "operating_threshold",
        "match_radius_px",
        "selection_protocol",
        "composite_score",
        "is_best",
    )
    for epoch in range(1, max_epochs + 1):
        row = {
            "epoch": epoch,
            **run_training_epoch(
                model,
                loader,
                device,
                optimizer,
                scaler,
                center_weight,
                mask_auxiliary_weight,
                epoch,
                satellite_target,
            ),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started,
            **{key: "" for key in validation_fields},
        }
        scheduler.step()
        if epoch % interval == 0:
            metrics, checkpoint_heatmaps = evaluate_full_images(
                model,
                validation_records,
                centers_by_id,
                config,
                device,
                retain_heatmaps=True,
            )
            if checkpoint_protocol(config) == "protocol-v1":
                checkpoint_radius = float(
                    config["point_evaluation"]["checkpoint_match_radius_px"]
                )
                operating_rows = threshold_radius_sweep(
                    validation_records,
                    centers_by_id,
                    checkpoint_heatmaps,
                    config,
                    match_radii=(checkpoint_radius,),
                )
                selected = operating_rows[0]
                for key in (
                    "point_precision",
                    "point_recall",
                    "point_f1",
                    "point_tp",
                    "point_fp",
                    "point_fn",
                    "predicted_center_count",
                    "true_center_count",
                    "fp_per_image",
                    "count_mae_all",
                    "count_mae_positive",
                    "zero_case_fp",
                    "positive_case_recall",
                    "localization_error",
                ):
                    metrics[key] = selected[key]
                metrics["operating_threshold"] = selected["threshold"]
                metrics["match_radius_px"] = selected["match_radius_px"]
                metrics["composite_score"] = checkpoint_score(metrics, config)
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": config,
                "operating_threshold": metrics["operating_threshold"],
                "match_radius_px": metrics["match_radius_px"],
                "selection_protocol": checkpoint_protocol(config),
                "selection_score": metrics["composite_score"],
            }
            torch.save(checkpoint, output / f"checkpoint_epoch{epoch:04d}.pth")
            row.update(metrics)
            improved = metrics["composite_score"] > best_score
            row["is_best"] = improved
            if improved:
                best_score = metrics["composite_score"]
                bad_checks = 0
                torch.save(checkpoint, output / "checkpoint_best.pth")
            else:
                bad_checks += 1
        append_row(output / "train_log.csv", row)
        print(row, flush=True)
        if epoch % interval == 0 and bad_checks >= patience_checks:
            print(
                f"early stopping at epoch {epoch}; best composite={best_score:.6f}",
                flush=True,
            )
            break

    checkpoint = torch.load(
        output / "checkpoint_best.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    final_output = output / "validation_evaluation"
    fixed_metrics, heatmap_cache = evaluate_full_images(
        model,
        validation_records,
        centers_by_id,
        config,
        device,
        retain_heatmaps=True,
        threshold_override=float(checkpoint["operating_threshold"]),
    )
    checkpoint_radius = float(config["point_evaluation"]["checkpoint_match_radius_px"])
    calibration = threshold_radius_sweep(
        validation_records,
        centers_by_id,
        heatmap_cache,
        config,
        match_radii=(checkpoint_radius,),
    )
    selected = calibration[0]
    checkpoint["calibrated_operating_threshold"] = selected["threshold"]
    checkpoint["calibration_match_radius_px"] = selected["match_radius_px"]
    torch.save(checkpoint, output / "checkpoint_best.pth")
    final_metrics, heatmap_cache = evaluate_full_images(
        model,
        validation_records,
        centers_by_id,
        config,
        device,
        output=final_output,
        retain_heatmaps=True,
        threshold_override=float(selected["threshold"]),
    )
    summary = {
        "best_epoch": checkpoint["epoch"],
        "checkpoint_selection_threshold": checkpoint["operating_threshold"],
        "calibrated_operating_threshold": selected["threshold"],
        **final_metrics,
    }
    write_rows(output / "best_summary.csv", [summary])
    write_rows(final_output / "threshold_calibration.csv", calibration)
    sweep = threshold_radius_sweep(
        validation_records, centers_by_id, heatmap_cache, config
    )
    write_rows(final_output / "threshold_radius_sweep.csv", sweep)
    print(summary, flush=True)
    print({"fixed_threshold_checkpoint_metrics": fixed_metrics}, flush=True)
    print({"selected_validation_threshold": selected}, flush=True)
    print({"best_validation_sweep": sweep[0]}, flush=True)


if __name__ == "__main__":
    main()
