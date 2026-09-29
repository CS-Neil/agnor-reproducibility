# Frozen protocol: candidate-floor sensitivity

Date fixed: 2026-09-24, before rerunning inference or inspecting any
candidate-floor sensitivity result.

## Scope

Use the six immutable seed-42 matched-formulation checkpoints. Re-run inference
without changing checkpoints, validation-selected operating thresholds, NMS,
or matching radius on:

- each model's original v1 held-out patient; and
- every eligible CCAgT v2 patient/slide unit D--O.

## Candidate floors and outcomes

Candidate floors are fixed at 0.05, 0.10, 0.20, 0.30, 0.40, and 0.50. Extract
local maxima once at 0.05 and obtain higher-floor sets only by score filtering.
For every model, patient/unit, and floor, report candidate-floor-limited AP,
maximum recall at that floor, candidate count, and candidates per image.

For v1, summarize paired center-minus-mask differences over the three held-out
patients. For v2, first pair center and mask within each fold and patient, then
average fold differences within each of the 12 patient/slide units. Patients,
not images or fold models, are the independent units.

## Alternative v2 point truth

At identical floors and at each model's frozen operating threshold, compute all
metrics with both the formal COCO bounding-box centers and the prespecified
area-weighted polygon centroids. This is sensitivity analysis only; it does not
replace the formal bbox-center results or select any setting.

No floor is chosen from the results. The analysis reports the complete fixed
grid, including settings that reduce recall or yield zero candidates.
