# Results: frozen M2b on holdout3 (EXPLORATORY; PREREG_holdout4.md section 8)

Run 8 Nov 2026 with eval_holdout4.py --test holdout3 (commit 93cae93), after the holdout4
pre-registration (9d90977) and while holdout4 was being generated. All 15 checkpoints verified against
their pre-registered SHA-256. Output: runs/paper_budget/gt_att/eval4_holdout3.json. (The .txt copy was
overwritten by an accidental second call, which the script refused; the JSON is intact.)

Why exploratory: holdout3 results motivated the search that led to M2b (RESULTS_gt_att_holdout3.md),
although M2b itself was selected on training data only (RESULTS_diag_gt.md) and never saw holdout3.
The confirmatory test is holdout4.

## Consistency check
M0, M1, M2 and M3 reproduce the earlier holdout3 evaluation exactly (for example regret at 1e9:
M1 1.1404, M2 1.1537; within-circuit R^2 M1 +0.1309): same frozen models, same data, same code.

## Fit (seed-averaged; no prediction clipped)
| model | R^2 within | Kendall tau_b |
|---|---|---|
| M0 | -0.563 | 0.243 |
| M1 | +0.131 | 0.252 |
| M2 | -0.483 | 0.249 |
| M2b | +0.147 | 0.338 |
| M3 | -0.092 | 0.083 |

## Geometric-mean regret (true m)
| budget | M0 | M1 | M2 | M2b | M3 | min-kappa2 |
|---|---|---|---|---|---|---|
| 1e3 | 1.0008 | 1.0012 | 1.0009 | 1.0009 | 1.0020 | 1.0008 |
| 1e4 | 1.0077 | 1.0075 | 1.0082 | 1.0057 | 1.0129 | 1.0054 |
| 1e5 | 1.0356 | 1.0295 | 1.0289 | 1.0254 | 1.0445 | 1.0232 |
| 1e6 | 1.0923 | 1.0794 | 1.0839 | 1.0656 | 1.1002 | 1.0727 |
| 1e7 | 1.1367 | 1.1246 | 1.1365 | 1.1011 | 1.1741 | 1.1446 |
| 1e8 | 1.1450 | 1.1378 | 1.1516 | 1.1100 | 1.2192 | 1.1844 |
| 1e9 | 1.1445 | 1.1404 | 1.1537 | 1.1107 | 1.2251 | 1.1902 |

## M2b vs M1 (the holdout4 Family A tests, applied here exploratorily)
| budget | ratio M2b/M1 | 95% CI | p | Holm p | different choices |
|---|---|---|---|---|---|
| 1e8 | 0.9756 | [0.9520, 1.0004] | 0.018 | 0.026 | 71/162 |
| 1e9 | 0.9740 | [0.9501, 0.9989] | 0.013 | 0.026 | 71/162 |
The H5 criteria (ratio <= 0.98 and Holm p < 0.05 at both budgets) are satisfied on holdout3. This is
not a confirmation (see above).

## Observations
1. The training-data effect transfers to larger circuits: 3.1-3.2% lower regret in cross-validation
   (6-14 qubits), 2.4-2.6% on holdout3 (16-20 qubits).
2. The gain is in ranking, not magnitude: Kendall tau_b 0.338 vs 0.252, within-circuit R^2 only
   +0.016 above M1. The Transformer reorders cuts within a circuit better than the linear model.
3. M2b has the lowest regret of all models from 1e4 to 1e9. Against the minimum-kappa2 rule it is
   6.3% lower at 1e8 and 6.7% lower at 1e9, and lower at 1e6 (1.0656 vs 1.0727), the first model
   to beat that rule at 1e6; at 1e5 the minimum-kappa2 rule is still slightly better (1.0232 vs 1.0254).
4. Of the gap between M1 (1.1404) and the true-attenuation ceiling (1.042) at 1e9, M2b closes about 30%.
5. Power for holdout4: an effect of 2.4-2.6% is only just beyond the pre-registered 2% margin, so H5
   can fail on holdout4 even if the effect is real; the CI will be reported either way.

## Thesis use (conditional on holdout4)
If H5 holds on holdout4: "a graph model adds decision-relevant ranking information beyond a calibrated
ESP model, when it starts from that calibration". The holdout3 numbers stay labelled exploratory.
Comparison with IBM find_cuts and Qdislib for M2b is still to be run (exploratory, holdout3).
