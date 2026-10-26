# Pre-registration: Delta-learning follow-up to RQ1 (Part A)

Project: QUEST-MPI (M.Sc. thesis, Sara Muhammed Alshehri, King Khalid University)
Part A written: 21 October 2026, before the run `rq1-delta`. Frozen by the git tag `prereg-delta-A`.
Any later change is recorded in Section 7 with its date and reason.

## 1. Status of the existing RQ1 result

The confirmatory RQ1 result is the single holdout3 evaluation of 16 October 2026 (`PREREG_holdout3.md`, tag `prereg-holdout3-B`). It is unchanged by this follow-up and is reported in the thesis whatever this follow-up shows. On holdout3 the primary model (gt_size-layout) was significantly better than random, minimum overhead and METIS, and not significantly different from GBM-layout or layout-ESP.

## 2. Motivation and its provenance

Two observations motivate this follow-up.

1. **Training data.** The labeller applies gate-local noise (Qiskit `NoiseModel.from_backend`), so the log fidelity of a cut is, to first order, the sum of per-gate log fidelities that layout-ESP computes. The primary model receives per-gate calibration values but not this sum, and its readout must reconstruct it through pooling. Only about 10% of label variance lies within circuits, and the ListNet loss on within-circuit z-scores gives nearly tied circuits the same spread as well-separated ones.
2. **Holdout3, post hoc.** The primary model exceeded layout-ESP on the clustered_random and qaoa_sbm families and fell below it on twolocal, increasingly with width.

Because observation 2 comes from holdout3, holdout3 cannot confirm any model developed here. A model adopted under Section 5 is evaluated once on a new set, holdout4 (Section 6).

## 3. Interventions

- **A. Delta-learning** (Ramakrishnan et al., 2015). Score = w · log ESP_layout + f(G), where w is learned and the last layers of f are initialised to zero, so that the untrained model ranks exactly as layout-ESP. The run asserts this before training.
- **B. Gap-weighted pairwise loss.** A RankNet logistic loss on raw labels with pair weights |F_i − F_j|, normalised per circuit. This is the smooth counterpart of the primary metric.
- **Fair learned baseline: GBM-Delta.** A least-squares scale c on within-circuit centred log F and log ESP, followed by gradient boosting on the residual, with score = c · log ESP + predicted residual.

## 4. Design

| Item | Specification |
|---|---|
| Data | Training corpus only (corpus_v2/train), with fragment-layout hardware features. Holdout2 is reported as exploratory. Holdout3 is not read. |
| Split | Train on 6–10 qubits; early stopping on a stratified 6–10 qubit validation split (seed 2026); judged on 12–14 qubits (sel_12_14) |
| Factorial | {gt_size, gt_delta} × {ListNet, gap-weighted pairwise}, 5 seeds per cell, early stopping on gap-weighted τ |
| Reference | gt_ref: the frozen recipe (gt_size, ListNet, early stopping on plain τ) |
| Baselines | Random, minimum overhead, METIS, layout-ESP, GBM-layout, GBM-Delta |
| Hyperparameters | Identical to the frozen recipe (d = 64, 4 heads, 3 layers, lr 3e-4, weight decay 1e-4, at most 80 epochs, patience 15, 8 circuits per step) |
| Primary metric | Gap-weighted Kendall τ, per circuit, seed-averaged |

## 5. Decision rule

The pre-declared candidate is **gt_delta-pair**. On sel_12_14 it is compared with layout-ESP, gt_ref and GBM-Delta by paired Wilcoxon signed-rank tests over circuits, Holm-adjusted over these three comparisons, α = 0.05.

- gt_delta-pair is adopted as the primary model for holdout4 if and only if it is significantly better than **both** layout-ESP and gt_ref.
- The comparison with GBM-Delta is reported whatever its outcome. If it is not significant, the improvement is attributed to the physics prior and not to the attention architecture.
- If gt_delta-pair is not adopted, no further model variant is tried on this question. The holdout3 result stands as the RQ1 result, and this follow-up is reported as a negative result.

Factorial main effects (A and B) and results by family and width are exploratory.

**Known limitation.** sel_12_14 was already used for the Part B selection of 16 October. Repeated selection on the same set can overfit it, which is a further reason why adoption leads to a fresh holdout4 and not to a claim.

## 6. Holdout4 (only if the model is adopted)

Part B of this document will be written before any holdout4 circuit is generated. It will specify:
- the same generator, widths (16, 18, 20), families, labelling scheme v3.1, eligibility and stopping rules as holdout3;
- seeds from 99001 upward;
- the final model trained on all 6–14 qubit circuits with 5 seeds, with weight hashes recorded before unblinding;
- the confirmatory tests: the adopted model against layout-ESP, GBM-Delta and gt_size-layout, Holm over three.

## 7. Deviations log

| Date | Change | Reason |
|---|---|---|
| 21 Oct 2026 | Run `rq1-delta` completed: gt_delta-pair not adopted (vs layout-ESP +0.043, Holm p = 0.57; vs gt_ref +0.087, Holm p = 2.4e-5; vs GBM-Delta +0.088, Holm p = 0.018). | Decision applied as written in Section 5. |
| 21 Oct 2026 | Exploratory learning-curve run added (`run_rq1_curve.py`): nested 25/50/75/100% subsets of the 6–10 q training circuits, judged on sel_12_14. | Its only purpose is to decide whether labelling more training circuits is worth the cost. It does not change the Section 5 decision. A model trained on an enlarged corpus would be a new model: it requires its own pre-registered selection rule and a fresh holdout4 before any claim, and sel_12_14 (already used three times) cannot serve as its confirmation. |
