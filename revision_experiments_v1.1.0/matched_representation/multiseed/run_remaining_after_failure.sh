#!/usr/bin/env bash
set -euo pipefail

project=/home/neil/cancer
python_bin="$project/.venv-revision-uv/bin/python"
root="$project/revision_experiments/matched_representation/multiseed"

cd "$project"
nice -n 10 "$python_bin" \
    revision_experiments/matched_representation/run_formal_queue.py \
    "$root/configs/mask_f0_s2026.yaml" \
    "$root/configs/center_f0_s2026.yaml" \
    "$root/configs/mask_f1_s2026.yaml" \
    "$root/configs/center_f1_s2026.yaml" \
    "$root/configs/mask_f2_s2026.yaml" \
    "$root/configs/center_f2_s2026.yaml" \
    --status "$root/queue_status_seed2026.json" \
    > "$root/queue_seed2026.log" 2>&1

"$python_bin" - "$root/AVAILABLE_RUNS_COMPLETE" <<'PY'
import datetime
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(
    datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n",
    encoding="utf-8",
)
PY
