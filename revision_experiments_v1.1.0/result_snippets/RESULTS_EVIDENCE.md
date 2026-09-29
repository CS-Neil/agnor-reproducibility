# Immutable revision-experiment evidence

This file is generated only after all six matched held-out tests and all
72 model-by-v2-patient evaluations have immutable completion markers.
Differences are center minus mask. Values are descriptive.

## Matched-formulation arm summary

| Arm | Independent patients | AP | F1 | FP/image | MAE+ |
|---|---:|---:|---:|---:|---:|
| center | 3 | 0.265 ± 0.228 | 0.284 ± 0.244 | 7.681 ± 10.522 | 2.400 ± 1.124 |
| mask | 3 | 0.124 ± 0.075 | 0.220 ± 0.135 | 1.721 ± 0.118 | 1.331 ± 0.339 |

## Matched center-minus-mask paired summary

| Metric | n | Mean difference | Sample SD |
|---|---:|---:|---:|
| Point AP | 3 | 0.141 | 0.156 |
| Point F1 | 3 | 0.064 | 0.125 |
| FP/image | 3 | 5.961 | 10.493 |
| MAE+ | 3 | 1.069 | 1.136 |

## CCAgT v2 center-minus-mask patient-level cohort summary

| Metric | Independent units | Mean difference | Sample SD |
|---|---:|---:|---:|
| Point AP | 12 | 0.079 | 0.060 |
| Point F1 | 12 | 0.043 | 0.030 |
| FP/image | 12 | 0.430 | 0.577 |
| MAE+ | 12 | 0.729 | 0.650 |

## CCAgT v2 patient-level arm means and paired differences

| Unit | Mask AP | Center AP | Δ AP | Mask F1 | Center F1 | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|
| D | 0.068 | 0.191 | 0.123 | 0.200 | 0.272 | 0.072 |
| E | 0.076 | 0.286 | 0.210 | 0.209 | 0.298 | 0.089 |
| F | 0.001 | 0.005 | 0.004 | 0.004 | 0.028 | 0.024 |
| G | 0.053 | 0.129 | 0.076 | 0.155 | 0.202 | 0.046 |
| H | 0.111 | 0.269 | 0.158 | 0.265 | 0.336 | 0.071 |
| I | 0.054 | 0.118 | 0.064 | 0.177 | 0.189 | 0.012 |
| J | 0.030 | 0.090 | 0.060 | 0.104 | 0.138 | 0.034 |
| K | 0.011 | 0.071 | 0.060 | 0.051 | 0.115 | 0.064 |
| L | 0.013 | 0.023 | 0.010 | 0.055 | 0.042 | -0.013 |
| M | 0.014 | 0.049 | 0.034 | 0.067 | 0.093 | 0.026 |
| N | 0.018 | 0.079 | 0.061 | 0.065 | 0.129 | 0.064 |
| O | 0.043 | 0.126 | 0.083 | 0.152 | 0.177 | 0.024 |
