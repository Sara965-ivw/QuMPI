# Pre-registration: transfer of the frozen models to a second device (fake_sherbrooke)

Committed before any holdout4 circuit is labelled on fake_sherbrooke. The commit that adds this file,
paper_budget/device.py, paper_budget/datagen/relabel_device.py and paper_budget/eval_transfer.py must
precede the commit that adds data/corpus_v2/holdout4_sherbrooke/.

## 1. Question
All models were trained and tested on labels from one calibration snapshot (fake_brisbane). The
proposal states that cut quality is governed by circuit architecture and device calibration. If the
models use calibration as an input rather than encoding fake_brisbane itself, the same frozen models,
given the calibration of another device, should still choose good cuts there.
Question: do the frozen models, trained only on fake_brisbane labels, keep their advantage on circuits
labelled on a different device whose calibration they receive as input?

## 2. Why this device, and what is not tested
fake_sherbrooke (qiskit-ibm-runtime, archived calibration snapshot) is an IBM Eagle processor like
fake_brisbane: 127 qubits, heavy-hex connectivity, ECR two-qubit gates. Its T1, T2, single- and
two-qubit gate errors and readout errors differ qubit by qubit. The test therefore changes the
calibration while keeping the architecture, gate set and topology, so a change in performance can be
attributed to calibration. Not tested: a different topology or gate set (fake_torino, Heron, CZ),
synthetic sweeps of single noise parameters, calibration drift of one device over time, and
execution on hardware. These are limitations of the test, stated with its result.

## 3. Design
- Circuits: the 156 circuits of the holdout4 analysis set (data/corpus_v2/holdout4_analysis_set.json).
- Candidates: the candidate cuts of each circuit's fake_brisbane file, rebuilt from their records. The
  relabeller stops if a rebuilt candidate differs in structure (cut type, gate cuts, wire cuts, widths,
  sampling overhead) or if the rebuilt circuit gives different ideal observable values (1e-9).
- Labels: the frozen labeller (label_plan, scheme v3.1) on make_device("fake_sherbrooke"), transpiler
  seed 1234, optimisation level 2, PYTHONHASHSEED=0; the oracle check on the same records as before.
  A candidate whose routed fragment exceeds the 13-qubit ceiling on fake_sherbrooke is dropped and
  counted.
- Analysis set: the circuits of the 156 that pass the filter of PREREG_holdout4.md section 3 on the
  fake_sherbrooke labels (>= 3 candidates, >= 1 informative observable, a layout-ESP for every
  candidate). The number of circuits and of dropped candidates is reported.
- Models: the 15 frozen checkpoints of PREREG_holdout4.md (SHA-256 verified), the frozen prior
  (alpha 0.0509, beta 0.1245), the frozen linear weights w and the clip a_min. Nothing is trained,
  refitted or tuned on fake_sherbrooke data.
- Inputs: layout features and layout-ESP computed from the fake_sherbrooke calibration and the
  fake_sherbrooke fragment layouts (paper_budget.device.use_device; no file in stage2_predictor/ is
  edited). Structural cut features (routing overhead, mapped two-qubit count, depths) come from the
  fake_sherbrooke records.
- Metric and decision rule: as PREREG_holdout4.md (geometric-mean regret against the best candidate,
  budget-aware rule with the true m, N = 1e3 ... 1e9).
- Code check before labelling: `eval_transfer --check-holdout4` must reproduce every regret of
  eval4_holdout4.json within 1e-6 with the evaluation code used here.

## 4. Hypotheses and tests
Paired tests on the per-circuit log regret ratio: Wilcoxon signed-rank, 95% bootstrap CI of the
geometric-mean ratio (10 000 circuit resamples, seed 2026), as in eval_holdout4.py.

Family T (primary, Holm over 2):
- T1: regret of M2b <= 0.98 x regret of M1 at 1e8 and 1e9, Holm p < 0.05 at both (the H5 criteria).

Family S (secondary, Holm over 4):
- S1: regret of M2b < regret of M0 at 1e8 and 1e9, Holm p < 0.05 at both (the learned correction adds
  to the ESP prior on a new device).
- S2: regret of M0 < regret of the minimum-overhead rule at 1e8 and 1e9, Holm p < 0.05 at both
  (calibration-based choice beats overhead-only choice on a new device).

Family R (replication of H2 and H3, descriptive): M2 vs M1 and M2 vs M3 at 1e8 and 1e9, Holm over 4.

Control C (exploratory, no adjustment): M0 and M2b evaluated on the same fake_sherbrooke labels with
the features and ESP computed from the fake_brisbane calibration (the wrong device). Ratio of regret
with the correct calibration to regret with the wrong one, at 1e8 and 1e9.
- C1 (prediction): the ratio is below 1 for M0 at 1e9 (the ESP prior uses the calibration).
  No directional prediction for M2b.

Descriptive: regret of every model at every budget beside the fake_brisbane values of
eval4_holdout4.json (same circuits); fit metrics; per-family and per-width regret; per-seed regret of
M2b; number of candidates dropped.

## 5. Expected outcomes and power
On holdout4 (fake_brisbane) the M2b/M1 ratio was 0.951 and 0.948 (upper CI 0.978 and 0.976). If the
learned correction carries calibration-independent structure, a similar ratio is expected; if it
partly encodes fake_brisbane, the ratio moves towards 1. Because the effect on holdout4 was only
2-3 percentage points beyond the 0.98 margin, T1 can fail even if part of the effect transfers.
A failure of T1 is reported as "transfer of the improvement over M1 not shown", not as evidence that
the models do not transfer; the CI is reported in every case.

## 6. Decision rules
- T1 met: "the improvement of M2b over M1 transferred to a second device with a different
  calibration, without retraining" is reported as a confirmatory result for that device.
- T1 not met, S1 or S2 met: the corresponding part of the selection rule is reported as transferring.
- C1 not met (ratio >= 1 or CI including 1): reported as "no evidence that the choice uses
  qubit-level calibration values", consistent with the RQ1 hardware ablation of 12 October 2026.
- Every hypothesis is reported as met or not met with its ratio, CI and p-value. Nothing is revised,
  relabelled or re-evaluated after the output of eval_transfer has been read.

## 7. Threats to validity
One second device of the same family; one calibration snapshot per device; the fake_brisbane-fitted
prior and weights are reused as they are (by design); candidates were generated once and are not
regenerated for the new device; the decision rule uses the true m; simulation, not hardware.

## 8. Order of operations (each step committed before the next)
1. This file and the code, with tests passing.
2. `eval_transfer --check-holdout4` (reproduction, output in the log).
3. `relabel_device --device fake_sherbrooke` (labels; prints structure and timing only).
4. Commit the labels, the manifest and the relabel log.
5. `eval_transfer --test holdout4_sherbrooke --device fake_sherbrooke`, once.
No MPI timing experiment (RQ2) runs on the same machine while step 3 runs.
