# RQ1 code walkthrough (map for thesis-level review)

This guide documents the existing code structure; it does not replace line-by-line review or claim a clean environment test.

## Data path

1. `stage2_predictor/datagen/corpus.py` creates split datasets and manifests. The pre-existing manifest and selection files must be treated as provenance artifacts, not regenerated during release preparation.
2. `stage2_predictor/datagen/quest_phase1/` contains circuit generators, cut-candidate construction, fragment simulation/reconstruction and noisy labelling. `mixed_cuts.py` is central to the mixed wire/gate cut path; `labeling_v2.py` connects candidate plans to the v2 label records.
3. `stage2_predictor/quest_phase2/data.py` rebuilds graph examples, attaches per-candidate features and returns the training objects. `hardware.py` derives calibration/layout features.
4. `quest_phase2/models.py` defines the Graph Transformer variants and ranking loss. `run_rq1.py` supplies shared training/scoring functions; `run_rq1_v2.py` implements the frozen recipe; `run_rq1_hw.py` selects model/features; `run_rq1_h3.py` freezes and evaluates the primary registered model.
5. `quest_phase2/baselines.py` holds the comparison methods; `metrics.py` defines ranking metrics and paired tests. `runs/rq1_h3/holdout3_tests.json` and the CSVs are recorded outputs, not recomputed results.

## Key methodological point

The model is evaluated on *within-circuit candidate ordering*. A higher score for one candidate than another does not by itself mean the score is a calibrated fidelity probability. The primary metric is gap-weighted Kendall ranking agreement, with registered paired tests and Holm correction against the specified baselines.

## Why the frozen directory matters

`run_rq1_h3.py --stage evaluate` expects the frozen checkpoint files and `partB_record.json` in the `--out` directory, and writes evaluation outputs there. Never point `--out` at `runs/rq1_h3/` for reproduction: that is the archival result directory. The new runner copies the frozen checkpoints into a unique `runs/reproduce_rq1_*` directory first.

## Suggested thesis explanation for one module

For each function reviewed, record (a) the scientific responsibility, (b) inputs/outputs and invariants, (c) mathematical expression being implemented, (d) assumptions and error conditions, (e) the corresponding unit test, and (f) what a negative/failed test would mean. That keeps code explanations grounded in the implementation rather than merely narrating syntax.
