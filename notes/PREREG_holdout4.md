# Pre-registration: confirmatory test on holdout4

Committed before holdout4 is generated. Every model is frozen: the SHA-256 of all 15 checkpoints is
listed below and in paper_budget/eval_holdout4.py (FROZEN), which refuses to run on any mismatch.
The code used is the parent commit of the commit that adds this file.

## 1. Questions
- Q-A (primary, new): does M2b (M1 + Graph Transformer residual) choose cuts with lower error at large
  shot budgets than M1 (ESP + linear correction) on fresh circuits?
- Q-B (replication of PREREG_gt_att.md): do H1-H4 of that pre-registration hold on fresh circuits?
History: M2 failed H2 on holdout3 (RESULTS_gt_att_holdout3.md); M2b was selected afterwards by the
pre-registered training-data diagnostics (RESULTS_diag_gt.md). M2b has never been evaluated on a holdout.

## 2. Frozen models (train_v31, split and closed-form M0/M1 of runs/paper_budget/gt_att/main)
prior a = 0.0509 + 0.1245 e; a_min 0.0219; 5 seeds per Transformer; predictions averaged over seeds.
| checkpoint | SHA-256 |
|---|---|
| main/M2_seed0.pt | 963a208c747a799babe9b52e6897cf78e8e8cd0a3d2334c8eb12792a5e8c4d61 |
| main/M2_seed1.pt | 29b64eda3ccb7b723a67804cc8a7ae4a21304d463dba6e53f48be3c53beb116f |
| main/M2_seed2.pt | e2a3e697adfa8e23f668d642261025cd8157dcac8356ae1e04b1f6777f2b6e3c |
| main/M2_seed3.pt | 30d1d62cfac59e4c1e6ae9e46239347e0cdd0975bc105c1ef2dfbe3d5238c1c8 |
| main/M2_seed4.pt | 5c055be1a59640ea1e136a6a76b15c3b6231a041a7c9ed389b94ffb104d29309 |
| main/M3_seed0.pt | c28fccd0f00eafd298bf18328525a62fd2b6a1eae9df0b1b41230a661e6531a3 |
| main/M3_seed1.pt | e335cf1784c170553648c578fa954cf50f1b3bb6409f75e8c765820910859031 |
| main/M3_seed2.pt | 9fd55c0c3526e829427b5d0733e4da06ab5bdcb1b26912f646300608d82ff66d |
| main/M3_seed3.pt | 4a401fedd5c66ac29e6e38402a49fdfc31c1ba2c712a4df1f9a8a14471d6a486 |
| main/M3_seed4.pt | a8f0241339459a4fb2cd51335b299a8e8dfaa6ecad515c21aa3e3be6afd7d5ba |
| m2b/M2b_seed0.pt | d4229b2846d2eef400214dff5e5c4f2c0a39d1baa37aaec1d170a4f62ab761af |
| m2b/M2b_seed1.pt | ba7e4d84d6a7deac26d7a54cd5cd61c3e3cc8add67218fc378af91e00e34ea03 |
| m2b/M2b_seed2.pt | caf23371276b20a97bb00fffec7f020a2088062eaa301efabdf7693252b3f38a |
| m2b/M2b_seed3.pt | 96723d3f1263ffd2c5dc7a4e9fb8cf5540e0ebdacff3604b17757e0b135667ec |
| m2b/M2b_seed4.pt | ece0c55833a2fcb0426760383a271640210b44f9d9ee86037159527e089f7eb9 |

## 3. Test set
holdout4: paper_budget/datagen/gen_holdout4.py = the holdout3 procedure (sequential sampling, 16 eligible
circuits per (width, family) cell or 60 seeds; widths 16, 18, 20; three families; holdout3 labelling
settings; scheme v3.1; PYTHONHASHSEED=0) with seeds 99,001-99,540, disjoint from every earlier split
(tested). Analysis set: every labelled holdout4 circuit with >= 3 candidates, >= 1 informative observable
and layout-ESP for every candidate (the filter used on holdout3). Labels are not summarised before the
evaluation; the generator prints structure only.

## 4. Metrics and tests
As PREREG_gt_att.md: seed-averaged predictions clipped to [a_min, -ln 0.01]; selection
argmin (1 - f)^2 m + (kappa2 - f^2 m)/N with the true m; regret = MSE(chosen)/MSE(best candidate);
geometric means over circuits; Wilcoxon signed-rank on per-circuit log regret ratios (zero differences
dropped, their number reported); 95% bootstrap CI (10,000 circuit resamples, seed 2026).
- Family A (Holm over 2): M2b vs M1 at 1e8 and 1e9.
- Family B (Holm over 4, as in PREREG_gt_att.md): M2 vs M1 and M2 vs M3 at 1e8 and 1e9.

## 5. Hypotheses
- H5 (primary): geometric-mean regret of M2b is at least 2% lower than M1 (ratio <= 0.98) at 1e8 AND
  1e9, with Holm-adjusted p < 0.05 at both budgets.
- H6 (descriptive): M2b within-circuit R^2 > M1 within-circuit R^2.
- H1-H4: unchanged from PREREG_gt_att.md, evaluated on holdout4.

## 6. Decision criteria
- H5 met: "starting the Graph Transformer from the calibrated ESP model (M2b) lowers the error of
  budget-aware cut selection at 1e8-1e9 shots on unseen circuits" is claimed, with the effect size and
  CI, and with the history in section 1 stated alongside.
- H5 not met: no decision benefit of the Graph Transformer is claimed; M1 is reported as the method.
- M2: by PREREG_gt_att.md section 5 an M2 claim needed H2 on holdout3 AND holdout4; holdout3 failed,
  so no M2 claim is possible whatever holdout4 shows. Family B is reported as replication.

## 7. Power (stated in advance)
On holdout3 the 95% CI of the M2/M1 regret ratio had a half-width of about 2.8% (162 circuits). The
cross-validated M2b effect is 3.1-3.2% (RESULTS_diag_gt.md), with a CI reaching 1.00. A true effect of
this size passes H5 roughly half of the time; a failure of H5 is therefore not proof of no effect, and
the CI is reported either way.

## 8. Exploratory (no claims)
M2b on holdout3 (eval_holdout4.py --test holdout3; the models are frozen above, so this cannot change
the holdout4 test); per-family results; seed spread; M2b vs M2, vs minimum-kappa2 and vs M0;
IBM find_cuts and Qdislib on holdout4, if they are run later.

## 9. Threats to validity
Training circuits 6-14 qubits vs test 16-20; 134 training circuits; untuned hyperparameters; one device
model (fake_brisbane); GPU training (RTX 5070 Laptop, driver 591.44) is not bitwise reproducible, which is
why the checkpoints, not the training runs, are frozen.
