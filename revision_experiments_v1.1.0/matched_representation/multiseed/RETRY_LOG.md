# Multiseed execution retry log

## 2026-09-25: `center_f2_s123`

The first attempt started at 2026-09-24T14:42:57Z and stopped during epoch 7
after the prespecified finite-gradient assertion detected a non-finite gradient
in `dec1.block.3.weight`. No validation checkpoint had yet been written, and no
held-out test inference was performed. The failure was not an out-of-memory or
service interruption.

Before inspecting any held-out result for this run, one clean retry was fixed.
The failed directory and log are retained unchanged under
`failed_attempts/center_f2_s123_attempt1`. The retry uses the identical frozen
configuration, code, data, seed, mixed-precision settings, learning rate, and
finite-gradient failure rule, and starts from random initialization rather than
from the incomplete trajectory. No hyperparameter is changed. If the same run
fails again, it will be reported as an unusable/non-finite seed trajectory; no
additional numerical workaround will be introduced into the matched analysis
without an explicitly amended protocol.

The clean retry started at 2026-09-25T14:27:26Z. It reproduced the same epoch
1--6 trajectory (up to floating-point rounding) and stopped at the same epoch-7
mini-batch with a non-finite gradient in the same parameter,
`dec1.block.3.weight`. The run is therefore classified as a reproducible
numerical-stability failure for this frozen seed/configuration. It is not
eligible for a held-out metric pair and will not be retried or repaired by
changing only this arm. The remaining prespecified seed-2026 runs continue so
that the failure does not suppress otherwise obtainable evidence.
