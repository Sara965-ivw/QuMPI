# Results: paper_budget/analysis/obs_structure.py on holdout3 (scheme v3.1), 2026-11-04

Exploratory design analysis. Holdout3 is not a test set for the paper.
Predictions were committed beforehand in notes/PREDICTIONS_obs_structure.md.
Data: 206 circuits, 90,899 (candidate, observable) pairs.

## A. Multiplicative damping
- |mu*| > 0.2 (n = 22,996): mu/mu* quantiles 5/25/50/75/95% = 0.587 / 0.846 / 0.912 / 0.940 / 0.969
- A1 (>= 80% of ratios in [0, 1.05]): observed 1.000 -> MET
- A2 (median mu > 0, single-Z with |mu*| < 0.02, n = 17,875): observed +0.0003 -> met in sign, negligible in size
- Share of total squared error by |mu*|: [0,0.02) 0.008; [0.02,0.2) 0.194; [0.2,0.6) 0.589; [0.6,1] 0.209
- Caveat: labels are exact density-matrix expectations (no shot noise) and readout confusion is
  symmetric (labeling_v2._apply_confusion), which is exactly multiplicative on Z-type observables.
  The finding holds for this noise model; it is not evidence about hardware.

## B. Light cone
- (circuit, observable) pairs with both kinds of candidates: 2,561
- median log10 error difference (in - out): +0.039 (about 1.09x squared error); share > 0: 0.549
- Wilcoxon one-sided p = 4.9e-4; pairs are not independent (206 circuits), so this p is optimistic
- B1 (median > 0 AND share >= 0.60): NOT MET
- Possible confounders not controlled: number of cuts, cut type

## Decisions (pre-registered rules)
- A: learning target becomes the attenuation f; an offset head o is kept as insurance against
  non-symmetric noise.
- B: fragment-graph conditioning (innovation 1) takes priority over light-cone conditioning.
- Any follow-up light-cone measure is post hoc and exploratory; confirmation is reserved for holdout4.
