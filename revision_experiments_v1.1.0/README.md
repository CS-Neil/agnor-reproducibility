# AgNOR revision-experiment archive v1.1.0

This archive supports the matched satellite-formulation experiment and the
nonoverlapping CCAgT v2 expanded-release evaluation added during revision of
manuscript Access-2026-42816. It includes the prespecified protocols, resolved
configurations, environment and run manifests, validation-selection records,
six immutable v1 test outputs, all 72 fixed-threshold v2 model-by-unit outputs,
11 successful additional-seed model outputs, both records from the reproducible
failed trajectory, eight available patient--seed pairs, candidate-floor and
point-truth sensitivity results, patient-level summaries, audit records,
analysis code, and SHA-256 checksums. The missing ninth multiseed pair is not
imputed and the available-pair summaries are explicitly descriptive.

The archive intentionally excludes provider images, masks, annotations,
derived satellite-center labels, model checkpoints, smoke-test outputs, and
per-field raster prediction exports, and the overlap-audit image montage.
Obtain CCAgT data from the providers cited in
the manuscript and comply with their licensing terms.

`manifest.json` records every archived file hash. This archive is distributed
with the tagged GitHub release `v1.1.0` of the AgNOR reproducibility repository;
the associated full Git commit identifies the exact repository state. No DOI is
claimed for this archive.
