#!/usr/bin/env python3
"""Continue the revision pipeline after downloads and formal training finish."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import csv
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MATCHED = PROJECT / "revision_experiments" / "matched_representation"
V2 = PROJECT / "revision_experiments" / "ccagt_v2"
STATUS = PROJECT / "revision_experiments" / "continuation_status.json"
TARGETS = ("mask", "center")
FOLDS = range(3)
WAIT_SECONDS = max(30, int(os.environ.get("REVISION_PIPELINE_POLL_SECONDS", "300")))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(stage: str, detail: str, extra: dict | None = None) -> None:
    payload = {
        "updated_at_utc": utc_now(),
        "stage": stage,
        "detail": detail,
        "execution_mode": "local_processes_only_no_codex_or_api_calls",
        "poll_interval_seconds": WAIT_SECONDS,
    }
    if extra:
        payload.update(extra)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(STATUS)


def run_logged(command: list[str], log: Path, environment: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as target:
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env=environment,
            stdout=target,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def formal_progress(paths: list[Path]) -> dict:
    completed = sum(path.is_file() for path in paths)
    progress: dict = {
        "formal_tests_completed": completed,
        "formal_tests_total": len(paths),
    }
    queue_status = MATCHED / "formal_queue_status.json"
    if not queue_status.is_file():
        return progress
    rows = json.loads(queue_status.read_text(encoding="utf-8"))
    active = next(
        (row for row in reversed(rows) if row.get("status") in {"training", "formal_test"}),
        None,
    )
    if not active:
        return progress
    output = Path(active["output"])
    progress.update(
        {
            "current_run": output.name,
            "current_run_stage": active["status"],
        }
    )
    train_log = output / "train_log.csv"
    if not train_log.is_file():
        return progress
    with train_log.open(newline="", encoding="utf-8") as source:
        epoch_rows = list(csv.DictReader(source))
    if not epoch_rows:
        return progress
    progress["current_epoch"] = int(epoch_rows[-1]["epoch"])
    validations = [row for row in epoch_rows if row.get("composite_score")]
    if validations:
        best_index = max(
            range(len(validations)),
            key=lambda index: float(validations[index]["composite_score"]),
        )
        best = validations[best_index]
        progress.update(
            {
                "latest_validation_epoch": int(validations[-1]["epoch"]),
                "latest_validation_composite": float(validations[-1]["composite_score"]),
                "best_validation_epoch": int(best["epoch"]),
                "best_validation_composite": float(best["composite_score"]),
                "validation_checks_since_best": len(validations) - best_index - 1,
                "early_stopping_patience_checks": 5,
            }
        )
    return progress


def wait_for(paths: list[Path], stage: str) -> None:
    while True:
        missing = [path for path in paths if not path.is_file()]
        if not missing:
            return
        extra = formal_progress(paths) if stage == "waiting_for_matched_training" else {}
        write_status(
            stage,
            f"{len(paths) - len(missing)}/{len(paths)} markers complete; "
            f"{len(missing)} remaining",
            extra,
        )
        time.sleep(WAIT_SECONDS)


def main() -> None:
    environment = os.environ.copy()
    baseline = str(PROJECT / "baseline")
    environment["PYTHONPATH"] = (
        baseline
        if not environment.get("PYTHONPATH")
        else baseline + os.pathsep + environment["PYTHONPATH"]
    )

    download_marker = V2 / "raw" / "download_complete.json"
    wait_for([download_marker], "waiting_for_v2_download")
    if not (V2 / "audit_results.json").is_file():
        write_status("v2_audit", "running extraction and content audit")
        run_logged(
            [sys.executable, str(V2 / "extract_and_audit.py")],
            V2 / "extract_and_audit.log",
            environment,
        )

    formal_markers = [
        MATCHED
        / "runs"
        / f"{target}_f{fold}_s42"
        / "formal_test"
        / "TEST_COMPLETE_DO_NOT_RERUN"
        for target in TARGETS
        for fold in FOLDS
    ]
    wait_for(formal_markers, "waiting_for_matched_training")
    matched_summary_marker = MATCHED / "summary" / "SUMMARY_COMPLETE"
    if not matched_summary_marker.is_file():
        write_status("matched_summary", "summarizing six immutable formal tests")
        run_logged(
            [sys.executable, str(MATCHED / "summarize_formal_results.py")],
            MATCHED / "summary" / "summarize.log",
            environment,
        )

    for target in TARGETS:
        for fold in FOLDS:
            name = f"{target}_f{fold}_s42"
            output = V2 / "evaluations" / name
            audit = json.loads((V2 / "audit_results.json").read_text(encoding="utf-8"))
            if not audit["evaluation_gate_passed"]:
                raise RuntimeError("v2 nonoverlap gate did not pass")
            complete = [
                output / patient_id / "EVALUATION_COMPLETE_DO_NOT_RERUN"
                for patient_id in audit["eligible_patient_slide_units"]
            ]
            if all(path.is_file() for path in complete):
                continue
            write_status("v2_fixed_evaluation", f"running {name}")
            run_logged(
                [
                    sys.executable,
                    str(V2 / "evaluate_v2_fixed.py"),
                    "--config",
                    str(MATCHED / "configs" / f"{name}.yaml"),
                    "--checkpoint",
                    str(MATCHED / "runs" / name / "checkpoint_best.pth"),
                    "--output",
                    str(output),
                ],
                V2 / "evaluation_logs" / f"{name}.log",
                environment,
            )

    v2_summary_marker = V2 / "V2_SUMMARY_COMPLETE"
    if not v2_summary_marker.is_file():
        write_status("v2_summary", "creating prespecified patient-level summaries")
        run_logged(
            [sys.executable, str(V2 / "summarize_v2_results.py")],
            V2 / "summarize_v2_results.log",
            environment,
        )
    snippets_marker = PROJECT / "revision_experiments" / "result_snippets" / "SNIPPETS_COMPLETE"
    if not snippets_marker.is_file():
        write_status("result_snippets", "rendering hash-locked Markdown and LaTeX evidence")
        run_logged(
            [
                sys.executable,
                str(PROJECT / "revision_experiments" / "render_result_snippets.py"),
            ],
            PROJECT / "revision_experiments" / "result_snippets" / "render.log",
            environment,
        )
    write_status("results_ready", "matched and v2 results are ready for manuscript update")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status("failed", repr(error))
        raise
