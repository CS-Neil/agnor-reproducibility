# Environment provenance correction (v1.0.1)

This documentation-only correction does not change the frozen splits,
configurations, predictions, result tables, or evaluation code.

## Evidence that applies to the formal methods

- The manuscript and archived profiling records identify an NVIDIA RTX 3090
  for training and profiling.
- The formal nnU-Net pipelines were configured as nnU-Net v2.8.1 2D plans.
- The custom C4 code imports PyTorch, NumPy, SciPy, and Pillow. Formal C4 YAML
  files record seed 42, batch size two, six data-loading workers, mixed
  precision, and FP32 computation for the center branch and loss.
- The nine formal configurations and the results are preserved in this package.

## Version-snapshot distinction

`pip_freeze.txt` and `release_manifest.json` under this directory were copied
from the later `C4_w020_centerFP32_final` artifact-preparation snapshot. That
snapshot records macOS/Python 3.11.4, NumPy 2.4.3, SciPy 1.17.1, and PyTorch
2.11.0. It is not evidence that these versions ran the formal GPU experiments.

An earlier Linux development-stage snapshot exists separately at
`releases/C4_w020_fp32_frozen/` in the project workspace. It records Python
3.12.3, PyTorch 2.8.0+cu128, NumPy 2.3.2, SciPy 1.16.1, and nnU-Net v2.8.1.
Its manifest marks the formal run as pending; these versions cannot be assigned
to all nine completed formal runs without the original server logs.

The exact per-run Python, PyTorch, CUDA, and cuDNN versions for all nine formal
GPU runs were not preserved in this package. This is an environment-provenance
limitation. If contemporaneous server logs are recovered, add them in a later
version with a source and timestamp for each run; do not relabel either snapshot
as a formal training environment.
