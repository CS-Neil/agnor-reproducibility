#!/usr/bin/env python3
"""Create the frozen additional-seed matched-formulation configurations."""

from __future__ import annotations

from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
MATCHED = HERE.parent
CONFIGS = HERE / "configs"
RUNS = HERE / "runs"
ADDITIONAL_SEEDS = (123, 2026)
TARGETS = ("mask", "center")
FOLDS = range(3)


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for seed in ADDITIONAL_SEEDS:
        for fold in FOLDS:
            for target in TARGETS:
                source = MATCHED / "configs" / f"{target}_f{fold}_s42.yaml"
                config = yaml.safe_load(source.read_text(encoding="utf-8"))
                name = f"{target}_f{fold}_s{seed}"
                config["experiment"]["name"] = f"matched_{name}"
                config["experiment"]["description"] = (
                    "Frozen post-review multiseed extension of the matched-backbone "
                    "satellite target/loss comparison"
                )
                config["experiment"]["release"] = (
                    "revision_matched_representation_multiseed_20260924"
                )
                config["training"]["seed"] = seed
                config["output"] = str(RUNS / name)
                destination = CONFIGS / f"{name}.yaml"
                destination.write_text(
                    yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
                )
                print(destination)


if __name__ == "__main__":
    main()
