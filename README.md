# QuMPI

**QuMPI: AI-driven quantum circuit cutting for MPI-distributed NISQ computing**

This release candidate packages the existing QUEST-MPI research implementation and its frozen RQ1 / preliminary RQ2 artifacts under the new repository name **QuMPI**. It is a research handoff and publication-preparation build, not evidence that every planned stage is implemented or that a fresh machine has already reproduced the results.

## Research status

| Component | Status in this package | What can be claimed |
|---|---|---|
| RQ1: calibration-aware candidate-cut ranking | **Evaluated** on a pre-registered holdout of 143 circuits at 16–20 qubits | The frozen model outperforms random, minimum-overhead, and METIS on the primary gap-weighted ranking metric; it was not shown statistically superior to GBM-layout or layout-ESP. |
| Finite-shot budget analysis | **Existing analysis included** | The preferred cut depends on shot budget in the analysed data; interpret according to its registered protocol and documented limitations. |
| RQ2: barrier-wait pilot | **Preliminary only** | Captured 576 timed runs across 24 training-split workloads, 4/8 ranks, four assignment policies and three repeats. Known timing/scheduler and cost-model limitations prevent treating this as the complete RQ2 study. |
| Stage 3: balance-aware QUBO | **Not implemented** | This package does not claim an implemented QUBO partitioner. |
| RQ3: end-to-end fidelity per wall time | **Not implemented** | There is no completed model-selected-cut → MPI execution → distributed reconstruction pipeline here. |

## Main RQ1 result (recorded, not rerun during packaging)

The frozen evaluation contains 143 selected circuits. The primary model `gt_size-layout` has a mean gap-weighted Kendall ranking score (`kendall_w`) of about **0.404**. Recorded pairwise differences versus random, minimum-overhead, METIS, GBM-layout, and layout-ESP are in `runs/rq1_h3/holdout3_tests.json`. The recorded Holm-adjusted p-values are approximately 1.3×10⁻¹⁷ vs random, 2.94×10⁻¹¹ vs minimum-overhead, 1.97×10⁻¹¹ vs METIS, 0.828 vs GBM-layout, and 0.693 vs layout-ESP. Non-significance is not proof of equivalence.

Use `runs/runs_ci_corrected_2026-10-25/METHOD.md` and its tables for the corrected circuit-level bootstrap intervals; do not silently substitute the historical row-level intervals in the original `runs/rq1_h3/summary*.csv`.

## Repository map

- `stage2_predictor/datagen/quest_phase1/`: circuit/cut candidates, fragment construction and noisy labels.
- `stage2_predictor/quest_phase2/`: graph construction, hardware/layout features, Graph Transformer, baselines and metrics.
- `stage2_predictor/run_rq1*.py`: RQ1 training, model selection, freezing and analysis drivers.
- `stage4_mpi/rq2_prelim.py`: preliminary MPI task-timing experiment.
- `stage4_mpi/rq2_analyse.py`: analysis of already-recorded RQ2 timings.
- `data/corpus_v2/`: frozen labelled corpus and split manifests used by the research runs.
- `runs/rq1_h3/`: frozen weights, weight-hash record, primary evaluation outputs and logs.
- `runs/rq2_prelim/`: raw timing records and analysis outputs from the pilot.
- `runs/rq1_hw/`, `runs/budget_h3/`, `runs/findcuts_compare/`, `runs/runs_ci_corrected_2026-10-25/`: supporting analyses.
- `runs_records/partB_record.json`: provenance record retained at the repository root.
- `data_manifest/`: file integrity manifests for the bundled data/artifacts and original source files.

The core source paths are intentionally retained from the original snapshot to minimise risk to import paths and historical reproducibility. Generated `__pycache__` and `.pytest_cache` files were excluded. The historical `quest.sh` was **not** copied into the repository; it had unsafe copy/commit/push/delete behaviour. `scripts/quest.sh` is a new, restricted runner.

## Integrity check

Run this read-only check from the repository root:

```bash
python3 scripts/verify_artifacts.py
```

It verifies hashes for the frozen corpus, model checkpoints, RQ1/RQ2 outputs, supporting results, and the root record against `data_manifest/FROZEN_ARTIFACTS_SHA256SUMS.txt`. It does not open holdout files for scientific analysis, run tests, retrain models or run simulations.

## Environment and experiments

See `environment/ENVIRONMENT.md` before installing dependencies. The source's root `requirements.txt` does not cover every package used in later experiment and analysis paths. A clean-install test has not yet been performed.

The following commands have different effects:

```bash
# Read-only integrity check; no experiments
./scripts/quest.sh verify-artifacts

# Analyse the existing RQ2 raw records in a NEW timestamped output folder.
# Does not run MPI simulations.
./scripts/quest.sh analyze-rq2

# RQ1 re-evaluation is an explicit opt-in and will re-read the already-unblinded
# holdout3 data. It writes only to a new runs/reproduce_rq1_* folder.
./scripts/quest.sh reproduce-rq1 --confirm
```




