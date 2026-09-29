#!/usr/bin/env python3
"""Download CCAgT v2 from its official per-file API with checksum checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


DATASET_ID = "wg4bpm33hj"
VERSION = 2
BASE = "https://data.mendeley.com"
EXPECTED_ARCHIVE_SHA256 = (
    "eb9f01e8feae0029056dac2dfa6f5e9780692e1a54c0df1bbd33cd279b9eb557"
)
USER_AGENT = "Mozilla/5.0 CCAgT-v2-revision-audit/1.0"


def fetch_json(url: str):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(item: dict, destination: Path, attempts: int = 20) -> dict:
    expected_size = int(item["content_details"]["size"])
    expected_hash = item["content_details"]["sha256_hash"]
    url = item["content_details"]["download_url"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size == expected_size:
        actual_hash = sha256(destination)
        if actual_hash == expected_hash:
            return {"path": str(destination), "status": "already_verified"}
        raise RuntimeError(f"Checksum mismatch for complete file: {destination}")
    if destination.exists() and destination.stat().st_size > expected_size:
        raise RuntimeError(f"Oversized partial file: {destination}")

    for attempt in range(1, attempts + 1):
        current_size = destination.stat().st_size if destination.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if current_size:
            request.add_header("Range", f"bytes={current_size}-")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = getattr(response, "status", response.getcode())
                if current_size and status != 206:
                    raise RuntimeError(
                        f"Server ignored Range for {destination} (HTTP {status})"
                    )
                mode = "ab" if current_size else "wb"
                with destination.open(mode) as target:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
        except Exception as error:
            if attempt == attempts:
                raise
            print(
                f"retry {attempt}/{attempts} {destination.name}: {error}",
                flush=True,
            )
            time.sleep(min(5 * attempt, 60))
            continue

        if destination.stat().st_size == expected_size:
            actual_hash = sha256(destination)
            if actual_hash != expected_hash:
                raise RuntimeError(f"Checksum mismatch: {destination}")
            return {"path": str(destination), "status": "downloaded_verified"}
        if destination.stat().st_size > expected_size:
            raise RuntimeError(f"Downloaded file is oversized: {destination}")

    raise RuntimeError(f"Download attempts exhausted: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent / "raw"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    snapshot = fetch_json(
        f"{BASE}/public-api/datasets/{DATASET_ID}/snapshot/{VERSION}"
    )
    archive = fetch_json(
        f"{BASE}/api/datasets-v2/datasets/{DATASET_ID}/zip?version={VERSION}"
    )
    if archive["sha256_hash"] != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("Official archive checksum changed unexpectedly")
    folders = fetch_json(
        f"{BASE}/public-api/datasets/{DATASET_ID}/folders/{VERSION}"
    )
    root_files = fetch_json(
        f"{BASE}/public-api/datasets/{DATASET_ID}/files?folder_id=root&version={VERSION}"
    )
    folder_files = {}
    jobs: list[tuple[dict, Path]] = []
    for folder in folders:
        files = fetch_json(
            f"{BASE}/public-api/datasets/{DATASET_ID}/files"
            f"?folder_id={folder['id']}&version={VERSION}"
        )
        folder_files[folder["name"]] = files
        for item in files:
            jobs.append((item, root / folder["name"] / item["filename"]))

    metadata = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "version": VERSION,
        "snapshot": snapshot,
        "archive": archive,
        "folders": folders,
        "root_files": root_files,
        "folder_files": folder_files,
    }
    (root / "official_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, item, destination): destination
            for item, destination in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)

    (root / "download_complete.json").write_text(
        json.dumps(
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": sorted(results, key=lambda row: row["path"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
