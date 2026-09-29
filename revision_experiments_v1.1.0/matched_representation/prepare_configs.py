"""Create the preregistered local mask-versus-center LOPO configurations."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "paper" / "reproducibility" / "configs"
CONFIGS = HERE / "configs"


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for fold in range(3):
        source = SOURCE / f"C4_w020_centerFP32_final_f{fold}_s42.yaml"
        with source.open(encoding="utf-8") as handle:
            base = yaml.safe_load(handle)
        for target in ("mask", "center"):
            config = yaml.safe_load(yaml.safe_dump(base))
            config["experiment"]["name"] = f"matched_{target}_f{fold}_s42"
            config["experiment"]["description"] = (
                "Reviewer-requested matched-backbone satellite-target ablation; "
                "separate from the frozen original formal release"
            )
            config["experiment"]["release"] = "revision_matched_representation_20260920"
            config["experiment"]["satellite_target"] = target
            config["data"]["root"] = str(ROOT / "dataset")
            config["center_labels"]["root"] = str(
                ROOT / "experiments" / "satellite_center_labels"
            )
            config["output"] = str(HERE / "runs" / f"{target}_f{fold}_s42")
            config["training"]["num_workers"] = 6
            config["training"]["max_epochs"] = 250
            config["training"]["iterations_per_epoch"] = 250
            config["training"]["early_stopping_patience_epochs"] = 50
            config["checkpoint_selection"]["interval_epochs"] = 10
            config["training"]["center_loss_weight"] = 0.2
            config["training"]["center_loss"] = (
                "binary_dice_bce_fp32"
                if target == "mask"
                else "modified_focal_fp32_logsigmoid"
            )
            config["training"]["mask_auxiliary_weight"] = 0.0
            config["point_evaluation"]["checkpoint_threshold"] = 0.30
            config["point_evaluation"]["ap_candidate_threshold"] = 0.05
            destination = CONFIGS / f"{target}_f{fold}_s42.yaml"
            with destination.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            print(destination)


if __name__ == "__main__":
    main()
