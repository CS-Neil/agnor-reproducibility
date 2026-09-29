# AgNOR segmentation reproducibility artifacts

This repository supports the manuscript *Rethinking Ultra-Sparse AgNOR
Satellite Segmentation: Patient-Independent Evaluation, Balanced Sampling, and
Point-Based Detection*.

The fixed `v1.1.0` release contains two complementary artifacts:

- `formal_v1/`: the v1.0.1 reproducibility package for the frozen original
  three-patient comparison;
- `revision_experiments_v1.1.0/`: matched mask-versus-center experiments,
  additional-seed results (including the documented failed trajectory), the
  eligible CCAgT v2 expanded-release evaluation, and candidate-floor and
  point-truth sensitivity analyses.

Use the tagged release and the full Git commit reported in the manuscript to
retrieve the exact version evaluated during resubmission. The release assets
also provide ZIP snapshots of both directories.

## Data and checkpoint exclusions

This repository does **not** redistribute provider images, masks, annotations,
derived satellite-center labels, model checkpoints, smoke-test outputs,
per-field raster prediction exports, or the overlap-audit montage. Obtain the
CCAgT data from the providers cited in the manuscript and comply with their
terms.

## Verification

Verify the original frozen artifact from the repository root with:

```bash
python formal_v1/evaluation/verify_artifact.py
python formal_v1/evaluation/recompute_summary.py
```

Each artifact includes a manifest with SHA-256 hashes and additional protocol
documentation. Environment provenance is described in
`formal_v1/environment/PROVENANCE.md` and in the revision-experiment run
manifests.

## License

Author-created source code and documentation in this repository are released
under the MIT License. Provider data are not included and remain governed by
their original terms. Generated result tables and manifests are supplied for
research verification with attribution to the associated manuscript.
