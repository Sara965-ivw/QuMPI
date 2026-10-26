# Corrected confidence intervals for RQ1 runs (25 Oct 2026)

## What was wrong
`quest_phase2/metrics.py` → `summarise` → `bootstrap_ci` resamples the **rows** of
`per_circuit.csv`. A model trained with 5 seeds contributes 5 rows per circuit, a
deterministic baseline (min_overhead, METIS, ESP) contributes 1. The bootstrap
therefore treats the 5 seeds of one circuit as 5 independent circuits
(pseudo-replication), and the intervals of seeded models are too narrow.

## What was changed (in this recomputation only; no repository file was modified)
1. Within each (model, split), the metric is averaged over seeds for each circuit.
2. Circuits are resampled with replacement: 2 000 resamples, `numpy.random.default_rng(0)`,
   percentile interval 2.5 / 97.5. Resample count, seed and percentile rule are the
   same as the original, so only the resampling unit differs.

Seed averaging per circuit is what `metrics.paired_test` already does before the
Wilcoxon test, so **no p-value and no pre-registered decision is affected**.
The point estimates (means) are unchanged up to floating-point rounding (≤ 1e-16),
because every circuit has the same number of seeds within a group.

## Validation of the input data
Before computing anything new, the script recomputes the ORIGINAL intervals from
`per_circuit.csv` and compares them with the stored `summary.csv`. The difference is
0.00e+00 for all eight runs that have a `summary.csv` (see `recompute_log.txt`).
The CSV files therefore contain every value needed; no model was retrained.

## Files
* `recompute_ci.py` — the script (reads `runs/*/per_circuit.csv`, writes only here)
* `<run>_ci_corrected.csv` — per (model, split): n_circuits, n_seeds, and for each
  metric the original mean/lo/hi, corrected mean/lo/hi and the width ratio
  (corrected width ÷ original width)
* `recompute_log.txt` — validation log
* Original files under `runs/` were read only and are unchanged.

## Main effect (gap-weighted τ, holdout3, frozen primary)
gt_size-layout: original [+0.375, +0.433] → corrected [+0.348, +0.462] (×1.97 wider).
gbm-layout: [+0.364, +0.420] → [+0.329, +0.459] (×2.30). Deterministic baselines are
unchanged. Seeded models' intervals widen by a factor of 1.0–2.5 (851 seeded model–split–metric groups; random excluded). The learning-curve reading rule (run_rq1_curve.boot_diff) already averages seeds per circuit and is unaffected.

## Decisions that used intervals
The width-selection rule in `run_rq1_v2.py` chose the simpler variant when the 95%
intervals of the best two overlapped. With the original intervals gt_hybrid and gt_size
already overlapped (gt_size selected); with the corrected, wider intervals they still
overlap, so the selection is unchanged. All other decisions (Part B, Delta, Card B,
holdout3 R1/R2) used seed-averaged paired Wilcoxon tests and are unaffected.

## How to reproduce
`python recompute_ci.py "<thesis papers>/runs" <output dir>` (numpy only, < 1 min).
