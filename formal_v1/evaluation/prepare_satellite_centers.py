#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from ccagt_baseline.data import load_pair, resolve_records
from ccagt_baseline.utils import load_config, save_yaml


def satellite_components(mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    labels, count = ndimage.label(mask == 3, structure=np.ones((3, 3), dtype=np.uint8))
    objects: list[dict] = []
    if not count:
        return labels.astype(np.int32), objects
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    centres = ndimage.center_of_mass(mask == 3, labels, range(1, count + 1))
    height, width = mask.shape
    for instance_id, ((row, column), area) in enumerate(zip(centres, areas[1:]), 1):
        instance = labels == instance_id
        touches_edge = bool(
            instance[0].any()
            or instance[-1].any()
            or instance[:, 0].any()
            or instance[:, -1].any()
        )
        objects.append(
            {
                "instance_id": instance_id,
                "row": float(row),
                "column": float(column),
                "area": int(area),
                "touches_edge": touches_edge,
                "in_bounds": bool(0 <= row < height and 0 <= column < width),
            }
        )
    return labels.astype(np.int32), objects


def gaussian_heatmap(
    shape: tuple[int, int], objects: list[dict], sigma: float, truncate: float
) -> np.ndarray:
    heatmap = np.zeros(shape, dtype=np.float32)
    radius = int(math.ceil(sigma * truncate))
    for item in objects:
        row, column = float(item["row"]), float(item["column"])
        peak_row = int(np.clip(round(row), 0, shape[0] - 1))
        peak_column = int(np.clip(round(column), 0, shape[1] - 1))
        row0 = max(0, peak_row - radius)
        row1 = min(shape[0], peak_row + radius + 1)
        col0 = max(0, peak_column - radius)
        col1 = min(shape[1], peak_column + radius + 1)
        yy, xx = np.ogrid[row0:row1, col0:col1]
        gaussian = np.exp(
            -((yy - peak_row) ** 2 + (xx - peak_column) ** 2) / (2 * sigma**2)
        )
        patch = heatmap[row0:row1, col0:col1]
        np.maximum(patch, gaussian.astype(np.float32), out=patch)
    return heatmap


def select_visualization_records(rows: list[dict], count: int) -> list[dict]:
    positive = [row for row in rows if int(row["satellite_count"]) > 0]
    edge = [row for row in positive if int(row["edge_satellite_count"]) > 0]
    close = [row for row in positive if float(row["minimum_center_distance"]) <= 8]
    large = sorted(positive, key=lambda row: int(row["satellite_count"]), reverse=True)
    chosen: list[dict] = []
    seen: set[str] = set()
    for group in (edge, close, large, positive):
        for row in group:
            if row["image_id"] not in seen:
                chosen.append(row)
                seen.add(row["image_id"])
            if len(chosen) >= count:
                return chosen
    return chosen


def save_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    heatmap: np.ndarray,
    objects: list[dict],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(mask == 3, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Satellite mask (n={len(objects)})")
    axes[2].imshow(image)
    if objects:
        axes[2].scatter(
            [item["column"] for item in objects],
            [item["row"] for item in objects],
            s=30,
            facecolors="none",
            edgecolors="yellow",
            linewidths=1.2,
        )
    axes[2].set_title("One centroid / component")
    axes[3].imshow(heatmap, cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("Gaussian max heatmap")
    for axis in axes:
        axis.axis("off")
    figure.savefig(path, dpi=120)
    plt.close(figure)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--truncate", type=float, default=3.0)
    parser.add_argument("--visualizations", type=int, default=30)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root is not None:
        config["data"]["root"] = str(args.data_root.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    heatmap_root = args.output / "heatmaps"
    visualization_root = args.output / "visualizations"
    heatmap_root.mkdir(exist_ok=True)
    visualization_root.mkdir(exist_ok=True)
    save_yaml(
        {
            "source_config": str(Path(args.config).resolve()),
            "sigma_px": args.sigma,
            "truncate_sigma": args.truncate,
            "overlap_rule": "pixelwise_max",
            "storage": "float16 compressed npz; mathematical range [0, 1]",
            "connectivity": 8,
        },
        args.output / "config.yaml",
    )

    records = resolve_records(config)
    center_rows: list[dict] = []
    image_rows: list[dict] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray, list[dict]]] = {}
    areas: list[int] = []
    failures: list[str] = []
    for record in tqdm(records, desc="satellite-centres"):
        image, mask = load_pair(record)
        labels, objects = satellite_components(mask)
        heatmap = gaussian_heatmap(mask.shape, objects, args.sigma, args.truncate)
        np.savez_compressed(
            heatmap_root / f"{record.image_id}.npz", heatmap=heatmap.astype(np.float16)
        )
        if not (0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0 + 1e-6):
            failures.append(f"range:{record.image_id}")
        if len(objects) != int(labels.max()):
            failures.append(f"one-centre-per-component:{record.image_id}")
        if any(not item["in_bounds"] for item in objects):
            failures.append(f"out-of-bounds:{record.image_id}")
        centres = np.asarray(
            [[item["row"], item["column"]] for item in objects], dtype=np.float32
        )
        if len(centres) > 1:
            distances = np.linalg.norm(centres[:, None] - centres[None, :], axis=2)
            distances[distances == 0] = np.inf
            minimum_distance = float(distances.min())
        else:
            minimum_distance = float("inf")
        for item in objects:
            areas.append(int(item["area"]))
            center_rows.append(
                {
                    "image_id": record.image_id,
                    "role": record.role,
                    **item,
                }
            )
        values, counts = np.unique(mask, return_counts=True)
        pixels = {int(value): int(count) for value, count in zip(values, counts)}
        image_rows.append(
            {
                "image_id": record.image_id,
                "role": record.role,
                "slide_id": record.slide_id,
                "patient_id": record.patient_id,
                "nucleus_pixels": pixels.get(1, 0),
                "cluster_pixels": pixels.get(2, 0),
                "satellite_pixels": pixels.get(3, 0),
                "satellite_count": len(objects),
                "edge_satellite_count": sum(item["touches_edge"] for item in objects),
                "minimum_center_distance": minimum_distance,
            }
        )
        if len(cache) < 100 and objects:
            cache[record.image_id] = (image, mask, objects)

    write_rows(args.output / "centers.csv", center_rows)
    write_rows(args.output / "image_summary.csv", image_rows)
    selected = select_visualization_records(image_rows, args.visualizations)
    for order, row in enumerate(selected, 1):
        record = next(record for record in records if record.image_id == row["image_id"])
        if record.image_id in cache:
            image, mask, objects = cache[record.image_id]
        else:
            image, mask = load_pair(record)
            _, objects = satellite_components(mask)
        heatmap = np.load(heatmap_root / f"{record.image_id}.npz")["heatmap"].astype(np.float32)
        save_visualization(
            image,
            mask,
            heatmap,
            objects,
            visualization_root / f"{order:02d}_{record.image_id}.png",
        )

    summary = {
        "images": len(records),
        "satellite_instances": len(center_rows),
        "positive_images": sum(int(row["satellite_count"]) > 0 for row in image_rows),
        "edge_instances": sum(int(row["edge_satellite_count"]) for row in image_rows),
        "visualizations": len(selected),
        "sigma_px": args.sigma,
        "truncate_sigma": args.truncate,
        "area_median": float(np.median(areas)) if areas else float("nan"),
        "area_mean": float(np.mean(areas)) if areas else float("nan"),
        "area_p95": float(np.percentile(areas, 95)) if areas else float("nan"),
        "automatic_check_failures": len(failures),
    }
    write_rows(args.output / "summary.csv", [summary])
    if failures:
        (args.output / "automatic_check_failures.txt").write_text("\n".join(failures))
        raise RuntimeError(f"Satellite center label checks failed: {failures[:10]}")
    print(summary)


if __name__ == "__main__":
    main()
