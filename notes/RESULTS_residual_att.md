# Results: residual attenuation model (residual_att.py)

Run 6 Nov 2026 on holdout3 (exploratory), commit 2029bdf; predictions committed before the run
(db6d7e8, notes/PREDICTIONS_residual_att.md). 162 circuits, 1726 candidates, GroupKFold(5).
Raw output: runs/paper_budget/residual_att/out.txt (not committed).

## R1. Out-of-fold within-circuit fit of the attenuation
| model | R^2 | median Spearman |
|---|---|---|
| lin_esp | -0.294 | +0.400 |
| lin_esp_feats | +0.114 | +0.409 |
| gbm_resid | -0.117 | +0.413 |

## R2. Geometric mean MSE / oracle (selection with the true m)
| budget | min_overhead | lin_esp | lin_esp_feats | gbm_resid | ceiling | gbm gap closed |
|---|---|---|---|---|---|---|
| 1e3 | 1.001 | 1.001 | 1.002 | 1.001 | 1.000 | - |
| 1e4 | 1.005 | 1.008 | 1.010 | 1.007 | 1.002 | 0.19 |
| 1e5 | 1.023 | 1.031 | 1.035 | 1.043 | 1.009 | -0.57 |
| 1e6 | 1.073 | 1.087 | 1.084 | 1.094 | 1.019 | -0.11 |
| 1e7 | 1.145 | 1.135 | 1.122 | 1.153 | 1.034 | -0.17 |
| 1e8 | 1.184 | 1.145 | 1.123 | 1.163 | 1.040 | -0.17 |
| 1e9 | 1.190 | 1.144 | 1.122 | 1.165 | 1.042 | -0.20 |

R3 (paired, Holm over budgets): no difference between lin_esp, lin_esp_feats and gbm_resid is
significant at any budget (smallest adjusted p 0.04, lin_esp_feats 0.1% worse at 1e3).

## Predictions
- P1 (lin_esp_feats R^2 in [0.08, 0.15]): met (0.114).
- P2 (gbm_resid R^2 >= lin_esp_feats + 0.03): not met (-0.117 vs 0.114).
- P3 (gbm_resid closes >= 25% of the gap at 1e8 and 1e9): not met (-0.17, -0.20: it widens it).
- P4 (gbm_resid never > 1% worse than lin_esp): not met (1.2% to 1.8% worse at 1e5 and 1e7-1e9).

Tally: 1 met, 3 not met.

## Interpretation
1. The pooled ESP slope overstates differences between cuts of the same circuit. Its
   within-circuit R^2 is negative (-0.294) although it ranks cuts moderately (Spearman +0.40):
   the slope fitted across circuits (about 0.10) is too steep within a circuit. lin_esp_feats,
   whose correction includes the demeaned ESP, can shrink it and reaches +0.114.
   This agrees with deployable_rule.py, where m_x0.25 (which also pulls the choice toward the
   minimum-overhead cut) was slightly better than m_true at 1e5-1e6. Hypothesis, not tested here:
   a slope fitted within circuits would calibrate f_hat.
2. GBM on these inputs did not learn the correction: worse than the linear correction in fit
   and in decisions. The available signal is small (R^2 about 0.1) and close to linear.
   With 1726 candidates from 162 circuits this is not evidence that no flexible model can learn
   it; sample size is a confounder (see next step).
3. At decision level the best simple model (lin_esp_feats) gains 2% at 1e8-1e9 (1.122 vs 1.144,
   22% of the gap to the ceiling), not significant. About 8% (1.122 vs 1.042) is not explained
   by ESP plus the 7 cut features.

## Decision (registered rule: P3 not met)
These features do not carry the decision-level signal on holdout3. The Graph Transformer is
justified only if graph structure after routing adds information. The baseline it must beat is
lin_esp_feats (the best simple model here), not the GBM.

## Next (proposed, to be pre-registered)
Remove the sample-size confounder: fit lin_esp, a within-circuit-slope ESP, lin_esp_feats and
gbm_resid on the training split (if it has scheme v3.1 labels) and evaluate on holdout3.

## Limitations
Candidates only; one attenuation per cut; holdout3 reused; GBM hyperparameters from
learnability.py, not retuned for the residual target.
