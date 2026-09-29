#!/usr/bin/env python3
"""Verify, extract, and audit the official CCAgT v2 per-slide archives."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RAW = HERE / "raw"
EXTRACTED = HERE / "extracted"
V1_ROOT = PROJECT / "dataset"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_pixel_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def difference_hash64(image: Image.Image) -> int:
    """Return a perceptual hash that is stable to ordinary JPEG re-encoding."""
    gray = np.asarray(
        image.convert("L").resize((9, 8), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    bits = (gray[:, 1:] > gray[:, :-1]).reshape(-1)
    value = 0
    for offset, bit in enumerate(bits):
        value |= int(bit) << offset
    return value


def comparison_thumbnail(image: Image.Image) -> np.ndarray:
    return np.asarray(
        image.convert("RGB").resize((64, 48), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def usable_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if member.is_dir() or "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        members.append(member)
    return members


def load_v1_hashes() -> tuple[
    dict[str, list[str]],
    dict[str, list[str]],
    list[tuple[str, int, np.ndarray]],
]:
    encoded: dict[str, list[str]] = defaultdict(list)
    pixels: dict[str, list[str]] = defaultdict(list)
    perceptual = []
    with (V1_ROOT / "metadata.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            path = V1_ROOT / row["image_path"]
            encoded[sha256(path)].append(row["image_id"])
            with Image.open(path) as image:
                pixels[decoded_pixel_hash(image)].append(row["image_id"])
                perceptual.append(
                    (
                        row["image_id"],
                        difference_hash64(image),
                        comparison_thumbnail(image),
                    )
                )
    return dict(encoded), dict(pixels), perceptual


def main() -> None:
    completion = RAW / "download_complete.json"
    if not completion.is_file():
        raise RuntimeError("Downloads are not complete; refusing to audit partial archives")
    official = json.loads((RAW / "official_metadata.json").read_text(encoding="utf-8"))
    coco = json.loads((RAW / "CCAgT_COCO_OD.json").read_text(encoding="utf-8"))
    images_by_slide: dict[str, set[str]] = defaultdict(set)
    coco_dimensions = {}
    coco_name_by_numeric_id = {}
    for image in coco["images"]:
        image_id = Path(image["file_name"]).stem
        slide = image_id.split("_", 1)[0]
        images_by_slide[slide].add(image_id)
        coco_dimensions[image_id] = (int(image["width"]), int(image["height"]))
        coco_name_by_numeric_id[int(image["id"])] = image_id

    expected = {
        (folder, item["filename"]): item
        for folder, items in official["folder_files"].items()
        for item in items
    }
    archive_checks = []
    for (folder, filename), item in sorted(expected.items()):
        path = RAW / folder / filename
        if path.stat().st_size != int(item["size"]):
            raise RuntimeError(f"Size mismatch: {path}")
        actual = sha256(path)
        wanted = item["content_details"]["sha256_hash"]
        if actual != wanted:
            raise RuntimeError(f"SHA-256 mismatch: {path}")
        archive_checks.append(
            {"path": str(path.relative_to(HERE)), "size": path.stat().st_size, "sha256": actual}
        )

    v1_encoded, v1_pixels, v1_perceptual = load_v1_hashes()
    v1_dhashes = np.asarray(
        [entry[1] for entry in v1_perceptual], dtype=np.uint64
    )
    encoded_overlap = []
    pixel_overlap = []
    perceptual_candidates = []
    likely_reencoded_overlap = []
    likely_transformed_overlap = []
    extracted_by_kind: dict[str, set[str]] = {"images": set(), "masks": set()}
    mask_modes = Counter()
    mask_colors = Counter()
    mask_high_color_files = []
    image_rows = []

    for kind in ("images", "masks"):
        for slide in sorted(images_by_slide):
            archive_path = RAW / kind / f"{slide}.zip"
            with zipfile.ZipFile(archive_path) as archive:
                members = usable_members(archive)
                member_stems = {Path(member.filename).stem for member in members}
                if len(member_stems) != len(members):
                    raise RuntimeError(f"Duplicate filename stems in {kind}/{slide}")
                if member_stems != images_by_slide[slide]:
                    missing = sorted(images_by_slide[slide] - member_stems)[:10]
                    extra = sorted(member_stems - images_by_slide[slide])[:10]
                    raise RuntimeError(
                        f"{kind}/{slide} COCO mismatch; missing={missing}, extra={extra}"
                    )
                for member in members:
                    image_id = Path(member.filename).stem
                    suffix = Path(member.filename).suffix or ".png"
                    destination = EXTRACTED / kind / slide / f"{image_id}{suffix}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    data = archive.read(member)
                    if destination.exists():
                        if destination.read_bytes() != data:
                            raise RuntimeError(f"Existing extraction differs: {destination}")
                    else:
                        with destination.open("wb") as target:
                            shutil.copyfileobj(io.BytesIO(data), target)
                    with Image.open(io.BytesIO(data)) as image:
                        if image.size != coco_dimensions[image_id]:
                            raise RuntimeError(
                                f"Dimension mismatch for {image_id}: "
                                f"{image.size} vs {coco_dimensions[image_id]}"
                            )
                        if kind == "images":
                            encoded_hash = hashlib.sha256(data).hexdigest()
                            pixel_hash = decoded_pixel_hash(image)
                            for v1_id in v1_encoded.get(encoded_hash, []):
                                encoded_overlap.append(
                                    {"v1_image_id": v1_id, "v2_image_id": image_id}
                                )
                            for v1_id in v1_pixels.get(pixel_hash, []):
                                pixel_overlap.append(
                                    {"v1_image_id": v1_id, "v2_image_id": image_id}
                                )
                            v2_dhash = difference_hash64(image)
                            distances = np.bitwise_count(
                                np.bitwise_xor(v1_dhashes, np.uint64(v2_dhash))
                            )
                            for candidate_index in np.flatnonzero(distances <= 4):
                                v1_id, _, v1_thumbnail = v1_perceptual[
                                    int(candidate_index)
                                ]
                                v2_thumbnail = comparison_thumbnail(image)
                                thumbnail_mae = float(
                                    np.abs(
                                        v1_thumbnail.astype(np.int16)
                                        - v2_thumbnail.astype(np.int16)
                                    ).mean()
                                )
                                candidate = {
                                    "v1_image_id": v1_id,
                                    "v2_image_id": image_id,
                                    "dhash_hamming_distance": int(
                                        distances[candidate_index]
                                    ),
                                    "thumbnail_rgb_mae": thumbnail_mae,
                                }
                                v1_serial = re.search(r"m(-?\d+)$", v1_id)
                                v2_parts = image_id.split("_")
                                candidate["field_serial_match"] = bool(
                                    v1_serial
                                    and len(v2_parts) > 1
                                    and v1_serial.group(1) == v2_parts[1]
                                )
                                perceptual_candidates.append(candidate)
                                if thumbnail_mae <= 5.0:
                                    likely_reencoded_overlap.append(candidate)
                                if candidate["field_serial_match"]:
                                    likely_transformed_overlap.append(candidate)
                            image_rows.append(
                                {
                                    "image_id": image_id,
                                    "patient_id": f"patient_{slide}",
                                    "slide_id": f"slide_{slide}",
                                    "image_path": str(destination.relative_to(HERE)),
                                    "mask_path": "",
                                    "width": image.width,
                                    "height": image.height,
                                    "encoded_sha256": encoded_hash,
                                    "decoded_pixel_sha256": pixel_hash,
                                }
                            )
                        else:
                            mask_modes[image.mode] += 1
                            colors = image.getcolors(maxcolors=1024)
                            if colors is None:
                                mask_high_color_files.append(image_id)
                            else:
                                for count, color in colors:
                                    mask_colors[str(color)] += int(count)
                    extracted_by_kind[kind].add(image_id)

    if extracted_by_kind["images"] != extracted_by_kind["masks"]:
        raise RuntimeError("Extracted image/mask identifiers do not pair one-to-one")

    mask_paths = {
        path.stem: path
        for path in (EXTRACTED / "masks").glob("*/*")
        if path.is_file()
    }
    for row in image_rows:
        row["mask_path"] = str(mask_paths[row["image_id"]].relative_to(HERE))
    image_rows.sort(key=lambda row: row["image_id"])
    with (HERE / "metadata.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)

    category_names = {int(row["id"]): row["name"] for row in coco["categories"]}
    category_counts = Counter(int(row["category_id"]) for row in coco["annotations"])
    centers = []
    for annotation in coco["annotations"]:
        if int(annotation["category_id"]) != 3:
            continue
        x, y, width, height = (float(value) for value in annotation["bbox"])
        image_id = coco_name_by_numeric_id[int(annotation["image_id"])]
        centers.append(
            {
                "image_id": image_id,
                "row": y + height / 2,
                "column": x + width / 2,
                "source": "coco_satellite_bbox_center",
                "annotation_id": annotation["id"],
            }
        )
    centers.sort(key=lambda row: (row["image_id"], int(row["annotation_id"])))
    with (HERE / "satellite_centers.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(centers[0]))
        writer.writeheader()
        writer.writerows(centers)

    overlapping_v2_images = {
        row["v2_image_id"]
        for row in (
            encoded_overlap
            + pixel_overlap
            + likely_reencoded_overlap
            + likely_transformed_overlap
        )
    }
    excluded_units = sorted(
        {f"patient_{image_id.split('_', 1)[0]}" for image_id in overlapping_v2_images}
    )
    all_units = [f"patient_{slide}" for slide in sorted(images_by_slide)]
    eligible_units = [unit for unit in all_units if unit not in excluded_units]
    evaluation_gate_passed = len(eligible_units) >= 3

    results = {
        "official_archives_verified": len(archive_checks),
        "archive_checks": archive_checks,
        "coco_images": len(coco["images"]),
        "unique_image_ids": len(extracted_by_kind["images"]),
        "patient_slide_units": len(images_by_slide),
        "images_per_slide": {
            slide: len(ids) for slide, ids in sorted(images_by_slide.items())
        },
        "coco_annotations": len(coco["annotations"]),
        "category_counts": {
            category_names[category]: count
            for category, count in sorted(category_counts.items())
        },
        "satellite_centers": len(centers),
        "mask_modes": dict(mask_modes),
        "mask_colors": dict(mask_colors),
        "mask_files_exceeding_1024_colors": mask_high_color_files,
        "v1_v2_exact_encoded_overlap": encoded_overlap,
        "v1_v2_exact_decoded_pixel_overlap": pixel_overlap,
        "v1_v2_perceptual_candidates_dhash_le_4": perceptual_candidates,
        "v1_v2_likely_reencoded_overlap": likely_reencoded_overlap,
        "v1_v2_likely_transformed_overlap": likely_transformed_overlap,
        "overlapping_v2_image_ids": sorted(overlapping_v2_images),
        "excluded_patient_slide_units": excluded_units,
        "eligible_patient_slide_units": eligible_units,
        "minimum_eligible_units": 3,
        "evaluation_gate_passed": evaluation_gate_passed,
        "notes": [
            "The official README reports O=262 images; COCO and archives are checked against O=263.",
            "Any v2 patient/slide unit with an exact, likely re-encoded, or serial-linked perceptually matched v1 field is excluded in full before inference.",
            "Likely re-encoded overlap is prespecified as 64-bit dHash distance <=4 and 64x48 RGB-thumbnail MAE <=5.",
            "Likely transformed overlap is prespecified as 64-bit dHash distance <=4 plus equality of the v1 terminal m field serial and the second underscore-delimited v2 field serial.",
            "Evaluation proceeds only when at least three completely nonoverlapping v2 patient/slide units remain.",
            "Satellite centers use the prespecified COCO bounding-box center and are not tuned from model outputs.",
        ],
    }
    (HERE / "audit_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
