# QuMPI architecture and scope

## Current implemented path

1. **Circuit and candidate preparation.** The data-generation package represents circuits, enumerates candidate cuts, builds fragments, and constructs training/holdout JSON records under `stage2_predictor/datagen/quest_phase1/` and `stage2_predictor/datagen/corpus.py`.
2. **Hardware-aware features.** `stage2_predictor/quest_phase2/hardware.py` and `data.py` form graph/candidate features, including device-derived calibration/layout information.
3. **Ranking model and baselines.** `quest_phase2/models.py` implements the Graph Transformer; `baselines.py` contains non-learned and simpler-learning baselines; `metrics.py` provides ranking metrics and paired tests.
4. **RQ1 experiment drivers.** `run_rq1.py`, `run_rq1_v2.py`, `run_rq1_hw.py`, and `run_rq1_h3.py` cover training, model selection, freezing and the recorded evaluation. The frozen weights and evaluation outputs are in `runs/rq1_h3/`.
5. **RQ2 MPI timing pilot.** `stage4_mpi/rq2_prelim.py` maps fragment simulation tasks to MPI ranks and records per-rank timing/wait fields. `rq2_analyse.py` analyses an existing JSONL timing record. The collected pilot artifacts are in `runs/rq2_prelim/`.

## Proposed but not implemented end-to-end

The repository does not implement the full five-stage QuMPI pipeline. In particular, a QUBO balance-aware partition selector (Stage 3), a complete cut-selection-to-MPI integrated execution study (main RQ2), and distributed reconstruction / end-to-end quality-per-wall-time evaluation (Stage 5 / RQ3) remain future work. An empty `stage3_qubo/` directory in the old snapshot is not an implementation.

## Important separation

The Graph Transformer currently provides a **within-circuit ranking score**, not a calibrated prediction of absolute fidelity. Do not describe it as a validated numeric fidelity predictor on the strength of the primary frozen RQ1 model.
