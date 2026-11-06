# Decision: objective for cut selection (departure from the proposal, approved)

Date recorded: 6 Nov 2026. Recorded by Sara Alshehri.

## What the proposal says
Candidate cuts are ranked by predicted fidelity (Hellinger fidelity of the reconstructed output).

## What changes
Cuts are chosen by the predicted error of the estimated observables at the user's shot budget N:
MSE_k(N) = b2_k + v_k / N (paper_budget/targets.py). The Graph Transformer predicts the attenuation
of each cut (residual on layout-ESP, notes/PLAN_gt_att.md); the fidelity-trained RQ1 model and its
holdout3 result remain in the thesis unchanged as the proposal's original objective.

## Why (evidence, all exploratory on holdout3)
- Fidelity ranking ignores the sampling overhead: the fidelity-trained Transformer has about 5x the
  error of IBM find_cuts at 1e3 shots (notes/RESULTS_compare_tools.md).
- A budget-aware rule beats find_cuts by 10-14% at 1e5-1e9 shots (same file).
- The rule works without the ideal values (notes/RESULTS_deployable_rule.md).

## Approval
Approved by the supervisor, Dr. Majed Alateeq (as reported by Sara, 6 Nov 2026).

## What stays frozen
All RQ1 code, checkpoints and results (stage2_predictor/, runs/rq1*, PREREG_holdout3.md,
PREREG_delta.md) are not modified. New work lives in paper_budget/ and runs/paper_budget/.
