# Pre-registration: holdout3 (second confirmatory test set)

Project: QUEST-MPI (M.Sc. thesis, Sara Muhammed Alshehri, King Khalid University)
Part A written: 12 October 2026, before any holdout3 circuit is generated or labelled.
Part A is frozen by the git tag `prereg-holdout3-A`. Any later change is recorded in Section 9 with its date and reason.

## 1. Why this test set exists

The fresh holdout2 (33 eligible circuits, 16 to 20 qubits) confirmed that the selected model ranks cuts better than random: Kendall τ = 0.242, Wilcoxon p = 3.4e-4. Two questions it could not settle:

- **Comparisons with METIS, minimum overhead and GBM.** These were underpowered. From the observed spread of per-circuit differences (SD ≈ 0.47), detecting Δτ ≈ 0.11 with 80% power at α = 0.05 needs about 120 to 140 circuits.
- **Budget dependence of the optimal cut.** An exploratory analysis on the training and seen-holdout labels (`runs/crossover_bias*/`, commit 7cee511) found it. It could not be assessed at 16 to 20 qubits, because 19 of 31 seen-holdout circuits offered no choice between overhead levels.

Holdout2 has now been inspected, so neither question can be confirmed on it. Holdout3 is generated after this document is frozen and is evaluated once.

## 2. Data generation

| Item | Specification |
|---|---|
| Widths | 16, 18, 20 qubits |
| Families | clustered_random, twolocal, qaoa_sbm. Generator code and parameter draws are identical to holdout2 (`corpus.build_spec`). |
| Seeds | 97001 upward, contiguous per (width, family) cell. Disjoint from train (10001+), holdout (90001+) and holdout2 (95001+). |
| Device model | fake_brisbane (qiskit-ibm-runtime 0.47.0, calibration 26 Feb 2025), optimisation level 2, transpiler seed 1234 |
| Labelling | Scheme v3.1: g ≤ 2 gate cuts, w ≤ 1 wire cut, fragment width ≤ 13 qubits after routing, at most 12 candidates per circuit (stratified as in holdout2), noiseless oracle on at least 10% of candidates and on every cut type |
| New in v3.1 | Exact expectation values of a fixed observable basket for the ideal circuit, the noisy uncut circuit (where simulable) and every candidate (Section 4) |
| Eligibility | At least 2 labelled candidates and at least 2 distinct sampling overheads. This is a structural rule and never reads a fidelity or expectation value. |
| Stopping rule | Per cell, circuits are labelled in seed order until 16 are eligible or 60 seeds have been used. The analysis set is the first 16 eligible circuits per cell in seed order (maximum 144). |
| Shortfall | A cell that cannot reach 16 within 60 seeds is not topped up from other cells. The shortfall is reported. Expected: qaoa_sbm cells at 18 and 20 qubits, given holdout2 yields of 1/5. |
| Expected size | About 125 to 144 circuits |

## 3. Blinding

- **Loaders refuse holdout3.** `quest_phase2.data.load_split` and `corpus_summary.py` raise an error on holdout3 unless `QUEST_UNBLIND=holdout3` is set. The guard is covered by a regression test.
- **Labelling prints counts only.** It prints candidate counts and timing, never a fidelity or expectation value.
- **One confirmatory run.** The environment variable is set only for the single run in Part B, after the model is frozen.

## 4. Observable basket and exact finite-shot error

For each circuit the basket contains:

- Z_i for every qubit;
- Z_i Z_j for every qubit pair that shares a two-qubit gate (for QAOA, the cost terms);
- the global parity Z_0 … Z_{n−1}.

For a candidate cut k with sampling overhead γ_k², each QPD sample of a Pauli-Z string takes the value ±γ_k. Let μ_kS be the expectation of the unclipped noisy quasi-distribution and μ*_S the ideal value. Then the exact mean squared error at N shots is:

MSE_kS(N) = (μ_kS − μ*_S)² + (γ_k² − μ_kS²) / N

The per-circuit error of cut k is the mean over the basket: MSE_k(N) = mean_S MSE_kS(N). The budget-optimal cut at N is argmin_k MSE_k(N).

The closed form is checked against Monte Carlo sampling in `tests/test_observables.py`.

## 5. Hypotheses and tests

The significance level is α = 0.05 (two-sided) unless stated otherwise. Confidence intervals are 95% bootstrap intervals over circuits: 10 000 resamples, seed 2026.

### Family R: RQ1 replication and baseline comparison

**Primary metric: gap-weighted Kendall τ, written τ_w.** Within a circuit, each candidate pair (i, j) is weighted by |F_i − F_j|, where F is the Hellinger fidelity label. τ_w is the weighted sum of +1 for a correctly ordered pair and −1 for a reversed pair, divided by the sum of weights. A model whose scores are constant within a circuit gets τ_w = 0.

The reason for this metric is that most errors on holdout2 involved pairs whose labels differed by less than 0.001. Such errors have no practical consequence, but they dominate the unweighted τ.

**Secondary metrics:** plain Kendall τ (for continuity with holdout2), top-1 regret and NDCG@3.

| ID | Hypothesis | Test |
|---|---|---|
| R1 | The primary model (Part B) has mean τ_w > the random-score baseline. | Paired Wilcoxon signed-rank test on per-circuit τ_w, with rank-biserial r |
| R2 | The primary model's τ_w differs from each of minimum overhead, METIS and GBM (amended 16 Oct 2026, before unblinding: + layout-ESP; see Section 9). | Paired Wilcoxon tests (four after the amendment), Holm-adjusted as one family |

Results are reported for each family and each width, but conclusions are drawn only from the pooled tests above.

### Family B: budget dependence (stage 3 motivation, RQ3)

These are estimated from exact MSE (Section 4), not from model predictions.

| ID | Hypothesis | Decision rule |
|---|---|---|
| B1 | The budget-optimal cut changes between N = 10³ and N = 10⁶ in a substantial share of circuits. | Supported if the lower 95% bound of the share is > 20% |
| B2 | At N = 10⁶, the minimum-overhead rule has MSE ≥ 1.1 × optimal in a substantial share of circuits. | Supported if the lower bound is > 20% |
| B3 | At N = 10³, the highest-fidelity rule has MSE ≥ 1.1 × optimal in a substantial share of circuits. | Supported if the lower bound is > 20% |

The 20% threshold was set as a practical-relevance floor, conservatively below the exploratory estimates of 33% to 48% on smaller circuits. The minimum-overhead rule is credited with the best cut among those tied at minimum overhead.

### Exploratory (labelled as such in the thesis)

- Results by family and width.
- Hellinger-ranked versus MSE-ranked optimal cuts.
- The hardware-feature variants: ESP baseline and fragment-layout calibration features.

## 6. Analysis rules

- **Single evaluation.** There is one evaluation run, and no model variant, metric or threshold is chosen after unblinding.
- **Complete reporting.** All hypotheses in Section 5 are reported whatever their outcome, including failures.
- **Failed circuits.** A circuit whose labelling fails after selection is excluded and listed with the reason. Model or metric performance is never a reason for exclusion.
- **Data-guided choices after unblinding.** Any such choice is reported as exploratory.

## 7. Part B: frozen primary model

Part B is completed before unblinding.

**Selection rule (fixed 16 Oct 2026, before the selection run `rq1-hw`).** Candidates are gt_size and gt_hybrid, each with calibration features from the uncut layout (as in holdout2) or from the fragment layouts (`hardware.py`). They are trained on 6–10 qubits with early stopping on a 6–10 qubit validation split, and judged on 12–14 qubits (training circuits only). The current primary, gt_size with uncut-layout features, is kept unless an alternative is significantly better on 12–14 qubits in gap-weighted τ: a paired Wilcoxon test over circuits (seed-averaged), Holm-corrected over the three alternatives, α = 0.05. If several alternatives qualify, the one with the highest mean is chosen. The chosen model is then trained on all 6–14 qubit circuits with 5 seeds, and its weights and commit are recorded below before `QUEST_UNBLIND` is set.

| Field | Value |
|---|---|
| Primary model ID | gt_size with fragment-layout hardware features (gt_size-layout) |
| Selected by | Section 7 rule, run `rq1-hw` on training circuits (runs/rq1_hw/partB_selection.json): all three alternatives were significantly better than gt_size-uncut; gt_size-layout had the highest mean |
| Code commit | recorded in runs/rq1_h3/partB_record.json and git tag prereg-holdout3-B |
| Config hash | SHA-256 of every weight file, in partB_record.json, verified before evaluation |
| Date | 16 Oct 2026 |

## 8. Part C: stage 3 policy evaluation

Part C is written before Phase 5 runs.

Stage 3 policies (the budget-derived objective, fixed-weight objectives of the MOSAIQC type, minimum overhead and METIS) are developed on train and seen-holdout data only. They are evaluated once on holdout3 by exact MSE at pre-declared budgets, and wall-time budgets are added once Phase 4 provides rank timings.

## 9. Deviations log

| Date | Change | Reason |
|---|---|---|
| 13 Oct 2026 | Scheduler queues only the number of circuits still needed per cell (was twice that number); runs with 4 workers instead of 6 | Round 1 queued 288 circuits for 144 slots and 6 workers caused memory pressure on 20-qubit circuits, giving an estimated 80+ h. The eligibility rule, the stopping rule and the selection rule (first 16 eligible per cell in seed order, at most 60 seeds) are unchanged, and no label was inspected. |
| 16 Oct 2026 | Labelling finished: 143 circuits selected; 20-qubit QAOA cell reached 15 of 16 at the 60-seed limit | As pre-registered, the short cell is reported and not topped up. No label was inspected. |
| 16 Oct 2026 | Part B selection rule written (Section 7) | Written before the selection run; it uses training circuits only. |
| 16 Oct 2026 | Family R2 extended: layout-ESP added as a fourth baseline, Holm over four; GBM uses fragment-layout features | On training circuits, layout-ESP was the strongest method (gap-weighted tau 0.491 on 12-14 qubits). Omitting the strongest available baseline would make R2 unfair. Decided before holdout3 was opened. |
| 16 Oct 2026 | Exploratory additions: placement-only controls (gt_size-layout_uniform and ESP with device-median calibration) | Separates the effect of fragment placement and SWAP distance from the effect of per-qubit calibration. Not part of any confirmatory test. |
| 16 Oct 2026 | Family R evaluated (run rq1-h3, Part B tag prereg-holdout3-B). Results: runs/rq1_h3/holdout3_tests.json | Single evaluation as pre-registered. |
| 16 Oct 2026 | Family B is evaluated in a second, separate run (`budget_h3.py`, tag prereg-holdout3-Bfam), after family R. Section 3 had foreseen a single run with QUEST_UNBLIND set. | The family B analysis code had not been written when family R was run. Family B uses no model; its budgets (10³, 10⁶), ratio (1.1), floor (20%) and rules were fixed in Part A. The family R run computed ranking metrics only; no expectation value or MSE of holdout3 was computed or inspected before this entry. |
| 16 Oct 2026 | Family B implementation details not stated in Part A, fixed before the family B run: (i) the optimal set at N contains every cut within a relative tolerance of 1e-9 of the minimum MSE, and B1 counts a change only when the optimal sets at 10³ and 10⁶ are disjoint; (ii) the highest-fidelity rule receives the same tie-breaking as the minimum-overhead rule (best MSE among tied cuts); (iii) negative per-shot variances, which can arise only numerically, are clipped at zero and counted; (iv) the lower bound is the 2.5th percentile of the bootstrap distribution, and a Wilson interval is reported as a sensitivity check | Each choice is conservative or symmetric between the rules, and none depends on holdout3 data. |
