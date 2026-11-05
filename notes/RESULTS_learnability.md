# Results: paper_budget/analysis/learnability.py on holdout3 (scheme v3.1), 2026-11-05

Exploratory design analysis; holdout3 is not a test set for the paper.
Predictions: notes/PREDICTIONS_learnability.md (committed before the run).
Data: 162 circuits, 1726 candidates, 5 folds grouped by circuit, GBM settings fixed in advance.

## A1. Out-of-fold prediction of the attenuation (within circuit)
- linear_esp:  R^2 -0.294, median Spearman +0.400
- gbm_esp:     R^2 -1.587, median Spearman +0.280
- gbm_esp_cut: R^2 -1.720, median Spearman +0.367
- P1 (gbm_esp_cut R^2 >= 0.30): NOT MET.
- Design flaw (acknowledged after the run): the models were trained on the raw attenuation,
  dominated by between-circuit differences that are irrelevant to selection; their
  within-circuit deviations are therefore noisy and R^2 is negative. Training on the
  within-circuit (demeaned) target would be the correct design. Spearman, which is
  insensitive to scale, also shows no gain over ESP, so the conclusion is unlikely to change.
  Any rerun with the corrected target is post hoc and exploratory.

## A2. Budget-aware selection, geometric mean of MSE / oracle MSE
budget  min_overhead  linear_esp  gbm_esp  gbm_esp_cut  true_att (ceiling)
1e3     1.001         1.001       1.001    1.001        1.000
1e4     1.005         1.008       1.013    1.007        1.002
1e5     1.023         1.031       1.038    1.032        1.009
1e6     1.073         1.087       1.086    1.089        1.019
1e7     1.145         1.135       1.171    1.137        1.034
1e8     1.184         1.145       1.177    1.135        1.040
1e9     1.190         1.144       1.181    1.139        1.042
- P2 (gbm_esp_cut <= 1.08 at 1e9): NOT MET (1.139).
- P3 (single-number ceiling <= 1.03 at every budget): NOT MET (max 1.042), but <= 1.05,
  so per-observable prediction is not required (it could gain at most about 4%).

## Interpretation
- Headroom available to any one-number-per-candidate predictor: about 0% up to 1e4,
  2% at 1e5, 5% at 1e6, about 10% at >= 1e7 (MSE ratio). No simple model captured any of it.
- The information needed is not in layout-ESP or the seven summary cut features.

## Decision (pre-registered rules)
P1 and P2 failed and R^2 < 0.15: prioritise C (candidate union); the residual
graph-transformer model is an exploratory bet. B (transpiled ESP) is folded into C,
since union candidates are transpiled for labelling anyway.
