# Prespecified CCAgT v2 expanded-release evaluation

Date fixed: 2026-09-20, before completion of matched-model training and before
inspection of any model output on CCAgT v2.

## Scope and entry gates

This is a same-institution, expanded-release evaluation, not multi-centre
external validation. CCAgT v2 contributes 15 patient/slide units (A--O), as
documented in the official README. Evaluation may start only if all official
archives pass their published SHA-256 checks, all 9,339 image/mask pairs and
COCO records reconcile, and content-level comparison identifies which, if any,
v2 patient/slide units reuse a v1 image. Any v2 unit containing a matched field
is excluded in its entirety before inference. Evaluation proceeds only if at
least three completely nonoverlapping patient/slide units remain. A failed gate
must be reported and stops the evaluation; it must not be worked around after
looking at model outputs.

The content audit uses both exact decoded-RGB pixel hashes and likely re-encoded
matches. The latter are prespecified as a 64-bit difference-hash Hamming
distance no greater than 4 together with mean absolute RGB error no greater
than 5 on a 64 x 48 thumbnail. Because the two releases may apply different
color processing, transformed-field overlap is also defined as difference-hash
distance no greater than 4 together with equality of the v1 terminal `m` field
serial and the second underscore-delimited v2 field serial. A match by any rule
excludes the containing v2 patient/slide unit. The exact/re-encoding thresholds
and whole-unit exclusion rule were fixed before extraction; the serial-linked
rule was added after the file audit exposed matching identifiers, but before
any inference on v2.

## Frozen models and thresholds

Evaluate all six models from the prespecified matched-representation experiment:
three v1 LOPO training folds for the mask formulation and their three paired
center-formulation runs. For every model, use `checkpoint_best.pth` selected
only on that run's v1 validation patient and the operating threshold calibrated
only on that same validation patient. Do not select a checkpoint, threshold,
NMS radius, matching radius, candidate floor, or model subset using v2 labels.

The three fold models are repeated trained-model measurements, not independent
patients and not an ensemble. Report every fold-by-eligible-v2-patient result.
For each eligible v2 patient and metric, first average the three paired fold-specific
center-minus-mask differences. The primary cohort summary is the mean and
sample standard deviation of the eligible patient-level paired differences. Also
report arm-specific patient values obtained by averaging the three fold models.
Do not treat individual fields or fold-by-patient rows as independent
statistical units.

## Labels and metrics

- Use COCO category 3 (`Satellite`) instances as the point truth. Define the
  point as the center of the published COCO bounding box; this rule is fixed
  before inference and does not use predictions.
- Preserve the v1 point protocol: local maxima, three-pixel NMS, five-pixel
  one-to-one matching, and candidate score floor 0.05 for point AP.
- Primary metric: candidate-floor-limited satellite point AP per patient.
- Secondary metrics at each model's frozen v1 validation threshold: point F1,
  precision, recall, FP/image, MAE on satellite-positive fields, overall count
  MAE, zero-case FP, positive-case recall, and localization error.
- Regional Dice is exploratory because v2 adds categories absent from the
  three-class training target. For compatibility only, map category 3
  (satellite) to the regional nucleus target as in the v1 C4 training code,
  retain categories 1 and 2 as nucleus and cluster, and map categories 4--7 to
  background. State this mapping wherever regional results appear.

## No-feedback rule

Inference is written to a new immutable output for each model and v2 patient.
A completion marker is written only after all images for that patient finish.
No v2 oracle threshold is part of the formal analysis. If an oracle diagnostic
is later computed, it must be stored separately, labeled post-hoc, and cannot
replace or influence the fixed-threshold results. Failures, zero detections,
and all completed models remain in the report.
