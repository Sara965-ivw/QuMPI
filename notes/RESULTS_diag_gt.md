# Results: stage-1 diagnostics of the attenuation Transformer (diag_gt_att.py)

Run 8 Nov 2026, 39.6 min on the RTX 5070; predictions committed before the run (0075096),
script 25f8b5d. Training data only (train_v31, 168 circuits, 5-fold StratifiedGroupKFold by family,
seed 2026), out-of-fold predictions, 3 seeds per Transformer. Raw: runs/paper_budget/diag_gt/
(diag.json, predictions.pkl). Exploratory.

## Out-of-fold results (within-circuit R^2 | Kendall tau_b | regret 1e8 | regret 1e9)
| model | 25% of data | 50% | 100% |
|---|---|---|---|
| M0 ESP | -0.077, 0.211, 1.269, 1.273 | -0.029, 0.211, 1.269, 1.273 | -0.031, 0.211, 1.269, 1.273 |
| M1 ESP + linear | +0.134, 0.283, 1.251, 1.258 | +0.115, 0.268, 1.263, 1.281 | +0.132, 0.291, 1.248, 1.253 |
| M2 ESP + Transformer | -0.067, 0.262, 1.228, 1.229 | -0.006, 0.280, 1.192, 1.188 | +0.017, 0.299, 1.232, 1.230 |
| M2b M1 + Transformer | +0.149, 0.309, 1.203, 1.203 | +0.152, 0.323, 1.208, 1.213 | +0.178, 0.350, 1.208, 1.214 |
| M3 direct Transformer | -0.104, 0.106, 1.348, 1.357 | -0.088, 0.107, 1.387, 1.398 | -0.063, 0.140, 1.381, 1.387 |

M2b / M1 regret at 100% data: 0.968 at 1e8 (95% CI [0.937, 1.000]), 0.969 at 1e9 (CI [0.937, 1.001]).

## Predictions
- D1 (M2 within R^2 rises < 0.05 from 50% to 100%): met (+0.023). From 25% to 100% it rises by
  +0.084 and tau_b by +0.037: a small, flattening data effect.
- D2 (M2b not better than M1: R^2 < M1 + 0.03 and regret within 1%): not met, in the direction that
  the decision rule names. M2b: R^2 +0.178 vs +0.132 (+0.046), tau_b 0.350 vs 0.291, regret 3.2% lower
  at 1e8 and 3.1% lower at 1e9. The bootstrap CIs reach 1.000 and 1.001: the decision gain is borderline.
- D3 (M3 tau_b rises >= 0.05 from 25% to 100%): not met (+0.034).
Tally: 1 met, 2 not met.

## Decision (pre-registered rule)
D2 crossed both thresholds (R^2 >= M1 + 0.03 and regret >= 2% lower at 1e8 and 1e9), so M2b becomes the
candidate model for the holdout4 pre-registration. Generating more training data is not justified by D1.

## Interpretation
1. After the linear calibration, the graph does carry some information: M2b improves on M1 in every
   metric at every data fraction (tau_b +0.03 to +0.06). Starting the network from M1 removes the
   calibration it failed to learn on its own (RESULTS_gt_att_holdout3.md, point 2).
2. Regret differences between fractions are noisy (for example, M2 at 50% has lower regret than at
   100%); within-circuit R^2 and tau_b are more stable and point the same way as the 100% regret.
3. The ESP miscalibration is much stronger on holdout3 than in cross-validation (M0 within R^2 -0.56
   vs -0.03): it grows with circuit size. Whether M2b keeps its advantage on 16-20 qubits is not
   tested here.
4. M3 is again the worst at decisions despite learning (tau_b 0.14).

## Limitations
Cross-validation on 6-14 qubit circuits only; M2b was conceived after the holdout3 results, so only a
fresh test set (holdout4) can confirm it; 3 seeds; untuned hyperparameters; CI upper bounds at 1.000.

## Next
Pre-register M2b vs M1 (and the original M2 vs M1) for holdout4; train M2b on the same train_v31
split as the main run; generate holdout4 with the holdout3 procedure and a new seed range.
