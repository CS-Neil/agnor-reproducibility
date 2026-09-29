#!/usr/bin/env python3
"""Run the prespecified matched-representation experiments and test once."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
import torch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
TRAIN_SCRIPT = HERE / "train_matched_representation.py"
TEST_SCRIPT = PROJECT / "baseline" / "evaluate_center_checkpoint.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_output(config_path: Path) -> Path:
    with config_path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    output = Path(config["output"])
    return output if output.is_absolute() else PROJECT / output


def run_logged(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def environment_snapshot() -> dict:
    packages = {}
    for name in ("torch", "numpy", "scipy", "Pillow", "PyYAML", "tqdm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "packages": packages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument(
        "--status",
        type=Path,
        default=HERE / "formal_queue_status.json",
    )
    args = parser.parse_args()

    config_paths = [Path(value).resolve() for value in args.configs]
    status_path = args.status.resolve()
    environment = os.environ.copy()
    baseline_path = str(PROJECT / "baseline")
    environment["PYTHONPATH"] = (
        baseline_path
        if not environment.get("PYTHONPATH")
        else baseline_path + os.pathsep + environment["PYTHONPATH"]
    )
    rows: list[dict] = []

    for config_path in config_paths:
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        output = read_output(config_path)
        marker = output / "formal_test" / "TEST_COMPLETE_DO_NOT_RERUN"
        row = {
            "config": str(config_path),
            "output": str(output),
            "status": "pending",
            "started_at_utc": None,
            "finished_at_utc": None,
            "train_exit_code": None,
            "test_exit_code": None,
        }
        rows.append(row)

        if marker.exists():
            row["status"] = "already_completed"
            write_json(status_path, rows)
            continue
        preflight_files = {"run_manifest.json", "environment.json", "pip_freeze.txt"}
        existing_files = (
            {path.name for path in output.iterdir()} if output.exists() else set()
        )
        if existing_files and not existing_files <= preflight_files:
            row["status"] = "blocked_existing_incomplete_output"
            write_json(status_path, rows)
            raise RuntimeError(
                f"Refusing to reuse incomplete/nonempty output directory: {output}"
            )

        output.mkdir(parents=True, exist_ok=True)
        row["status"] = "training"
        row["started_at_utc"] = utc_now()
        write_json(
            output / "run_manifest.json",
            {
                "created_at_utc": row["started_at_utc"],
                "python_executable": sys.executable,
                "config": str(config_path),
                "sha256": {
                    "config": sha256(config_path),
                    "training_script": sha256(TRAIN_SCRIPT),
                    "test_script": sha256(TEST_SCRIPT),
                },
                "no_feedback_rule": (
                    "The held-out test split is evaluated once after checkpoint "
                    "selection and threshold calibration on validation data."
                ),
            },
        )
        write_json(output / "environment.json", environment_snapshot())
        installed = sorted(
            {
                f"{distribution.metadata['Name']}=={distribution.version}"
                for distribution in importlib.metadata.distributions()
                if distribution.metadata.get("Name")
            },
            key=str.casefold,
        )
        (output / "pip_freeze.txt").write_text(
            "\n".join(installed) + "\n", encoding="utf-8"
        )
        write_json(status_path, rows)

        row["train_exit_code"] = run_logged(
            [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)],
            output / "training.log",
            environment,
        )
        if row["train_exit_code"] != 0:
            row["status"] = "training_failed"
            row["finished_at_utc"] = utc_now()
            write_json(status_path, rows)
            raise RuntimeError(f"Training failed for {config_path}")

        test_output = output / "formal_test"
        row["status"] = "formal_test"
        write_json(status_path, rows)
        row["test_exit_code"] = run_logged(
            [
                sys.executable,
                str(TEST_SCRIPT),
                "--config",
                str(config_path),
                "--checkpoint",
                str(output / "checkpoint_best.pth"),
                "--split",
                "test",
                "--output",
                str(test_output),
            ],
            output / "formal_test.log",
            environment,
        )
        if row["test_exit_code"] != 0:
            row["status"] = "formal_test_failed"
            row["finished_at_utc"] = utc_now()
            write_json(status_path, rows)
            raise RuntimeError(f"Formal test failed for {config_path}")

        marker.write_text(utc_now() + "\n", encoding="utf-8")
        row["status"] = "completed"
        row["finished_at_utc"] = utc_now()
        write_json(status_path, rows)


if __name__ == "__main__":
    main()
