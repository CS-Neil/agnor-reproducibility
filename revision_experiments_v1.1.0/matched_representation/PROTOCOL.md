# Prespecified matched satellite-target ablation

Date fixed: 2026-09-20, before training or inspecting any new held-out result.
Purpose: address Reviewer 1's concern that the original Balanced–C4 comparison
does not isolate target representation. This is a **new revision experiment**;
the frozen `releases/formal_lopo_final` results remain unchanged.

## Paired design

- Three LOPO folds retain the original train/validation/test patient assignments.
- Both arms use the existing C4 four-level U-Net encoder/decoder, base width 16,
  the same three-class regional head and the same one-channel satellite head.
- Both arms use the same training image pools, 35/35/20/10 sampling probabilities,
  1536 by 1024 patch, batch size two, augmentations, SGD with momentum 0.99,
  Nesterov acceleration, weight decay 3e-5, and initial learning rate 0.01.
- Seed 42, up to 250 epochs of 250 iterations, validation every 10 epochs,
  early-stopping patience 50 epochs, and the same polynomial learning-rate
  schedule apply to both arms.
- Both arms select a checkpoint on the validation patient by
  0.20 nucleus Dice + 0.20 cluster Dice + 0.60 candidate-floor-limited point AP;
  both subsequently select the operating threshold maximizing validation point
  F1 on 0.10--0.90 in 0.05 steps. The held-out patient is inferred once.
- Both arms use local-maximum extraction, NMS radius three pixels, five-pixel
  one-to-one point matching, and candidate score floor 0.05 for point AP.

The sole intended intervention is the satellite target/loss pair. The **mask**
arm supervises a binary satellite mask with Dice+BCE; the **center** arm
supervises a Gaussian center heatmap with modified focal loss. Each arm uses
the same one-channel satellite output and loss weight 0.20. The mask/center
target and corresponding loss cannot be separated by this design; conclusions
will refer to the combined satellite formulation, not representation alone.

## Reporting and no-feedback rule

Report every held-out patient separately and mean +/- SD across patients.
Primary metric: candidate-floor-limited satellite point AP. Secondary metrics:
validation-selected point F1, FP/image, MAE+ and regional Dice. The primary
comparison is paired center-minus-mask for all three held-out patients.
Post-hoc test-oracle thresholds, if computed, must be labeled diagnostic and
cannot replace validation-selected results. Do not change architecture,
hyperparameters, threshold grid, or selection rule after viewing a held-out
result. Report failures and completed runs alike.

## Practical execution

The current local machine has an NVIDIA GeForce RTX 4060 Ti with 16 GB VRAM,
31 GiB RAM, and 927 GiB free on the experiment volume. The original formal
weights are deleted, so both arms train from scratch locally. Installation
and a short smoke test precede the six formal runs; smoke tests use separate
outputs and never contribute to reported results. Exact package versions,
script SHA-256, configuration SHA-256, runtime logs, checkpoints, and
test-completion markers must be archived in this new experiment directory.
