# AgNOR LOPO reproducibility package v1.0.1

This artifact supports the frozen results in *Rethinking Ultra-Sparse AgNOR
Satellite Segmentation: Patient-Independent Evaluation, Balanced Sampling, and
Point-Based Detection*. It contains no patient images, labels, or model
checkpoints. Obtain the public CCAgT data from the provider named in the paper
and comply with its terms.

## What is reproducible from this package

- the exact three-fold patient/slide assignments;
- all frozen Standard, Balanced, and C4 configurations;
- satellite center-label generation and point-decoding definitions;
- dataset-level point AP and one-to-one distance matching;
- formal per-image/per-patient tables and validation-threshold transfer plots;
- manuscript-result consistency checks.

The held-out predictions and summary CSVs are frozen historical artifacts. The
post-hoc test-oracle sweep is diagnostic only and must not replace formal
validation-selected results.

## Layout

```text
configs/       frozen YAML configurations for all methods and folds
splits/        patient/slide metadata and LOPO assignments
evaluation/    point decoding, metrics, and formal analysis code
results/       frozen aggregate and per-image CSV outputs
environment/   preparation-environment snapshot and provenance clarification
manifest.json  artifact version, protocol, and file checksums
```

## Minimal verification

From the repository root:

```bash
python paper/reproducibility/evaluation/verify_artifact.py
python paper/reproducibility/evaluation/recompute_summary.py
```

The first command verifies every SHA-256 entry and the leakage-safe fold map.
The second regenerates `formal_lopo_summary_recomputed.csv` from the frozen
nine-row formal result table without reading test images or checkpoints.

The checksum verifier uses only the Python standard library. Summary
regeneration also uses only the standard library. The `environment/pip_freeze.txt`
and `environment/release_manifest.json` files describe the computer used to
prepare this artifact, not the exact GPU environment of all formal runs. See
`environment/PROVENANCE.md` before using any version number. Full training or
prediction replay also requires the source images and model checkpoints, which
are not redistributed here.

## Formal protocol

Each fold uses one patient/slide for training, one for validation, and one for
held-out testing. Checkpoint selection and detection-threshold calibration use
validation data only. Each held-out patient is evaluated once after freezing.

## GitHub release

This directory is distributed with the tagged GitHub release `v1.1.0` of the
AgNOR reproducibility repository. The release tag fixes the packaged artifact;
the associated full Git commit identifies the exact repository state. No DOI is
claimed for this artifact.
