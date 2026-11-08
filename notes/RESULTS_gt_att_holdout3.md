# Results: residual attenuation Transformer on holdout3 (exploratory)

Run 7-8 Nov 2026. Pre-registration notes/PREREG_gt_att.md (commit 62cb865, before training).
Training: runs/paper_budget/gt_att/main, git 62cb865, RTX 5070 Laptop, 5.5 min for 10 models.
Training split 134 circuits, validation 35; prior a = 0.0509 + 0.1245 e; a_min 0.0219.
Evaluation: eval_gt_att.py run once, 162 holdout3 circuits; output eval_holdout3.json / .txt.
No prediction was clipped.

## Fit of the attenuation (seed-averaged predictions)
| model | RMSE | MAE | R^2 | RMSE within | R^2 within | Kendall tau_b |
|---|---|---|---|---|---|---|
| M0 ESP prior | 0.1129 | 0.0783 | 0.682 | 0.0234 | -0.563 | 0.243 |
| M1 ESP + linear correction | 0.1118 | 0.0781 | 0.688 | 0.0174 | +0.131 | 0.252 |
| M2 ESP + Transformer residual | 0.1128 | 0.0782 | 0.683 | 0.0228 | -0.483 | 0.249 |
| M3 direct Transformer | 0.0979 | 0.0425 | 0.761 | 0.0195 | -0.092 | 0.083 |

## Geometric-mean regret (MSE of chosen cut / best candidate, true m)
| budget | M0 | M1 | M2 | M3 | min-kappa2 |
|---|---|---|---|---|---|
| 1e3 | 1.0008 | 1.0012 | 1.0009 | 1.0020 | 1.0008 |
| 1e4 | 1.0077 | 1.0075 | 1.0082 | 1.0129 | 1.0054 |
| 1e5 | 1.0356 | 1.0295 | 1.0289 | 1.0445 | 1.0232 |
| 1e6 | 1.0923 | 1.0794 | 1.0839 | 1.1002 | 1.0727 |
| 1e7 | 1.1367 | 1.1246 | 1.1365 | 1.1741 | 1.1446 |
| 1e8 | 1.1450 | 1.1378 | 1.1516 | 1.2192 | 1.1844 |
| 1e9 | 1.1445 | 1.1404 | 1.1537 | 1.2251 | 1.1902 |

## Primary tests (ratio = gm regret M2 / reference; Holm over the four tests)
| comparison | budget | ratio | 95% CI | p | Holm p | circuits with different choices |
|---|---|---|---|---|---|---|
| M2 vs M1 | 1e8 | 1.0121 | [0.9852, 1.0409] | 0.454 | 0.908 | 93/162 |
| M2 vs M1 | 1e9 | 1.0117 | [0.9849, 1.0408] | 0.462 | 0.908 | 92/162 |
| M2 vs M3 | 1e8 | 0.9446 | [0.9106, 0.9783] | 8.1e-4 | 2.7e-3 | 130/162 |
| M2 vs M3 | 1e9 | 0.9417 | [0.9073, 0.9759] | 6.8e-4 | 2.7e-3 | 130/162 |

## Hypotheses
- H1 (M2 within-circuit R^2 > M1): not met (-0.483 vs +0.131).
- H2 (M2 regret >= 2% lower than M1 at 1e8 and 1e9, Holm p < 0.05): not met. M2 is 1.2% worse,
  not significant. The lower CI bounds (0.9852, 0.9849) are above 0.98: a 2% improvement is
  excluded at 95% confidence.
- H3 (M2 better than M3 at 1e8 and 1e9): met (5.5% and 5.8% lower regret, Holm p 0.0027).
- H4 (M2 never > 1% worse than M0): met (largest ratio 1.008 at 1e9).
Tally: 2 met, 2 not met.

## Decision (pre-registered, section 5)
H2 failed on holdout3. M1 (ESP + linear correction) is reported as the best attenuation predictor;
the Graph Transformer result is negative. holdout4 is still run for the frozen models; no redesign is
evaluated on holdout4 without a new pre-registration.

## Interpretation
1. The Transformer residual barely moved away from ESP. Within-circuit R^2 went from -0.563 (M0) to
   -0.483 (M2); validation loss improved by under 1% per seed. Zero initialisation and early stopping
   kept it close to its starting point, as designed (H4 met), but it did not learn the correction.
2. The learnable error is mostly a calibration error. ESP's slope, fitted across circuits, overstates
   differences between cuts of one circuit (R^2 within -0.56). One linear term on the demeaned ESP
   fixes most of it (M1, +0.13). With 134 training circuits the Transformer did not recover this
   simple rescaling from graph inputs. The same pattern appeared in Step 8 (residual_att.py).
3. Pooled fit and decision quality disagree. M3 has the best pooled fit (R^2 0.76, MAE 0.043, against
   0.68 and 0.078 for M0): it predicts each circuit's overall attenuation better than the ESP prior
   extrapolated from 6-14 to 16-20 qubits. Its ranking within circuits is the worst (tau 0.08), and so
   are its decisions (+22% regret at 1e9). Cut selection depends on within-circuit differences;
   pooled metrics are not a valid proxy for it.
4. The ESP prior is essential for the Transformer (H3): without it, regret is 5-6% higher.
5. Exploratory (no claim): M1 has 4% lower regret than the minimum-kappa2 rule at 1e8-1e9
   (1.1378 vs 1.1844; 1.1404 vs 1.1902) and is within 1% of M0. At <= 1e6 the minimum-kappa2 rule
   is best; the predicted attenuation makes the budget rule switch to cleaner cuts too early there.
6. Transfer from small to large circuits: M1 trained on train_v31 reaches 1.140 at 1e9; the same model
   cross-fitted on holdout3 itself reached 1.122 (Step 8); the ceiling is 1.042.

## Threats to validity
holdout3 reuse (exploratory); 134 training circuits; frozen RQ1 hyperparameters, not tuned for this
target (by design); GPU training not bitwise reproducible (seed spread in eval_holdout3.json);
width shift 6-14 to 16-20 qubits.

## Thesis and paper use
Results chapter: honest negative result for the Transformer on the budget-aware target, with H3 as a
positive ablation (the ESP prior matters). Discussion: point 3 (pooled fit vs decision quality) and
point 2 (the learnable signal is a calibration). The comparison with IBM find_cuts and Qdislib
(section 6 of the pre-registration) is still to be run.
