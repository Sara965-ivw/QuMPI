# Predictions: frozen models vs METIS, random, IBM find_cuts and Qdislib on holdout4

Written and committed before find_cuts or Qdislib is run on any holdout4 circuit.
Status: exploratory, as listed in PREREG_holdout4.md section 8. No confirmatory claim follows from
this comparison; the predictions below are recorded so that the outcome cannot be reinterpreted
afterwards. Script: paper_budget/analysis/compare_holdout4.py (run once).

## 1. Why this comparison is run on holdout4
- M2b was selected after the holdout3 results were known (RESULTS_diag_gt.md), so a comparison
  of M2b on holdout3 would favour M2b. holdout4 is the only test set on which M2b is unselected.
- The proposal's RQ1 baselines are random placement, METIS and the overhead-minimising search of
  qiskit-addon-cutting (find_cuts). Qdislib is added because it was the strongest external
  baseline on holdout3 (RESULTS_compare_tools.md) and leaving it out would bias the comparison.
- CutQC is not included: its objective (subcircuit width and classical post-processing cost)
  differs from the error objective compared here, and its cut search requires a commercial MIP
  solver. The omission is reported as a limitation.

## 2. Fixed elements
- Models: the 15 frozen checkpoints of PREREG_holdout4.md (SHA-256 verified); nothing is trained.
  M2b, M1 and M0 choose with the budget-aware rule and the true m, exactly as eval_holdout4.py.
- Circuits: the 156 circuits of the analysis set (data/corpus_v2/holdout4_analysis_set.json,
  written by paper_budget.holdout4_list). A circuit whose find_cuts labelling times out or fails is
  removed from every method and listed (October rule). A circuit without a usable Qdislib result
  is removed from the Qdislib tests only.
- Tools: unchanged procedures of October and November. find_cuts: findcuts_label.py (width 13,
  seed 1234, decisions A1-A7, 1800 s per circuit). Qdislib 1.0.1: decisions Q1-Q5 and amendments
  1-2 of DECISIONS_qdislib.md (default find_cut, five seeds averaged, optimistic bound for
  unlabelled seeds). Same labeller, device (fake_brisbane), transpiler seed and scheme v3.1 as
  every other label; PYTHONHASHSEED=0.
- Error: MSE_k(N) = b2_k + v_k / N at N = 1e3 ... 1e9. Oracle: best of the candidates, the
  find_cuts cut and the labelled Qdislib cuts. Because the oracle includes the tools' cuts, the
  MSE/oracle values are not the regrets of eval4_holdout4.json (whose oracle is the best candidate).
- Subsets: all circuits; connected (find_cuts needs a cut); separable (find_cuts needs no cut).
- Tests: per circuit L = log(MSE_reference / MSE_method); geometric-mean ratio exp(mean L), > 1
  means the method has the lower error; Wilcoxon signed-rank with Holm over the 7 budgets; 95%
  bootstrap CI (10 000 circuit resamples, seed 2026).
- Reproduction check: before any tool result is read, the script recomputes the regrets of M0, M1
  and M2b and stops if they differ from eval4_holdout4.json by more than 1e-6.

## 3. Predictions
- P1 (find_cuts, large budgets). On connected circuits, M2b has a lower error than find_cuts at
  1e8 and 1e9: ratio find_cuts / M2b >= 1.10 with Holm p < 0.05 at both budgets.
  Basis: on holdout3 (connected), the minimum-overhead rule beat find_cuts by 11-16% from 1e5 to
  1e9 and the fidelity-trained model by 15% at 1e9; on holdout4, M2b's regret is 6.7% below that
  of the minimum-overhead rule at 1e9.
- P2 (Qdislib, large budgets). On connected circuits, M2b has a lower error than Qdislib at 1e8 and
  1e9: ratio Qdislib / M2b > 1 with Holm p < 0.05 at both budgets.
  Basis: on holdout3, Qdislib and the minimum-overhead rule were equal at 1e9 (1.17 and 1.17) and
  Qdislib was 10-12% better than find_cuts from 1e6. The expected ratio is about 1.05; the power
  to detect it is uncertain, so a CI that includes 1 leaves P2 undecided rather than refuted.
- P3 (small budget). On all circuits at 1e3, find_cuts and Qdislib both have a lower error than
  M2b: ratio < 1 with Holm p < 0.05 for both.
  Basis: the candidate set lacks the tools' lowest-overhead cuts, including the no-cut option of
  separable circuits; on holdout3 the union oracle was 1.90 times lower than the best candidate
  at 1e3 (RESULTS_compare_tools.md).
- P4 (proposal baselines). On all circuits, M2b has a lower error than METIS and than random
  placement at 1e8 and 1e9: ratio > 1 with Holm p < 0.05 for both, at both budgets.
  Basis: RQ1 on holdout3 (Holm p 2e-11 against METIS and 1e-17 against random, ranking metric).

## 4. Reported without a prediction
- D1: the smallest budget at which M2b has a significantly lower error than find_cuts (connected).
- D2: headroom of the candidate set, best candidate / union oracle, per budget and subset.
- D3: excluded circuits, Qdislib circuits without a result, and circuits with the optimistic bound.
- The same tables for M1, M0 and the minimum-overhead rule; per family and width in per_circuit.json.

## 5. Decision rules
- P1 met: "on unseen 16-20-qubit circuits, budget-aware selection with M2b has a lower error than
  IBM find_cuts at 1e8-1e9 shots" is reported, with the ratio and CI, as an exploratory result.
- P3 met: reported as a limitation of the candidate generator (not of the predictor): at small
  budgets the tools' lower-overhead cuts are better than any candidate.
- Any prediction not met is reported as not met, with the observed ratio and CI. No prediction is
  revised and the comparison is not rerun after its output has been read.

## 6. Threats to validity
One simulated device (fake_brisbane); widths 16-20 only; the tools' cuts are scored by our
labeller, not by their own reconstruction; Qdislib unlabelled seeds receive an optimistic bound,
which favours Qdislib; find_cuts and Qdislib are used with default settings and a 13-qubit
fragment limit, the same limit as our candidate generator.
