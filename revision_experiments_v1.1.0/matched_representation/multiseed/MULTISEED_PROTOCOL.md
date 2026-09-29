# Frozen protocol: matched-formulation multiseed extension

Date fixed: 2026-09-24, before training or inspecting any result from the two
additional seeds.

## Purpose

Estimate whether the matched mask-versus-center findings obtained with seed 42
are stable to training randomness. This is a post-review robustness extension,
not a replacement for the immutable six-run seed-42 comparison.

## Runs

- Seeds: 42, 123, and 2026.
- The completed seed-42 runs remain immutable and are reused.
- New runs: seeds 123 and 2026, both target/loss arms, and all three LOPO folds
  (12 new training runs).
- Queue order within each seed and fold: mask, then center.

Apart from `training.seed`, experiment name, release identifier, and output
path, every configuration field is identical to its paired seed-42 run. Each
run independently selects its checkpoint and reporting threshold using only
the corresponding validation patient, followed by one held-out test inference.
No held-out result may alter training, selection, or evaluation settings.

## Outcomes and independent units

The same AP, fixed-threshold point, false-positive, counting, localization, and
regional metrics are retained. Seed repetitions are repeated trained-model
measurements, not additional patients. Report:

1. every target × fold/patient × seed result;
2. paired center-minus-mask differences within fold/patient and seed;
3. for each held-out patient, the mean and sample standard deviation across
   the three seed-paired differences;
4. across the three independent held-out patients, the mean and sample standard
   deviation of the patient-level seed means;
5. direction counts across the nine repeated patient-seed pairs, explicitly
   labeled descriptive rather than nine independent clinical units.

No patient-level significance test is planned because only three independent
v1 patients are available.

## Completion rule

The extension is complete only when all 12 new runs have immutable held-out
test completion markers, the existing six seed-42 markers remain present, all
18 summaries are finite where defined, and the multiseed summary manifest
records hashes for every source summary, split, and checkpoint.
