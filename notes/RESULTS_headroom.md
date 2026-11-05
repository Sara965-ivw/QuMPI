# Results: paper_budget/analysis/headroom.py on holdout3 (scheme v3.1), 2026-11-05

Exploratory design analysis; holdout3 is not a test set for the paper.
Predictions: notes/PREDICTIONS_headroom.md (committed before the run).
Note: this run and obs_structure used paper_budget/targets.py before it was committed;
its tests reproduce budget_h3 to 1e-12, so the definitions are unchanged.

Data: 206 circuits, 2012 candidates, 0 without layout-ESP; Q1-Q3 use 162 circuits
(>= 3 candidates and at least one observable with |mu*| > 0.2).

## Q1. Attenuation explained by layout-ESP (within circuit)
- per-circuit Spearman(-log ESP, attenuation): median +0.400, share > 0: 0.778
- pooled within-circuit R^2 (in-sample): 0.054
- H1 (Spearman > 0 AND R^2 in [0.2, 0.8]): NOT MET. ESP ranks moderately but does not
  estimate magnitude.
- Calibration is stable across folds: attenuation ~ 0.01-0.02 + 0.10 * (-log ESP).

## Q2. Learnable remainder (2-fold by circuit, out-of-fold, linear)
- R^2 ESP only 0.049; ESP + 7 cut features 0.112; gain +0.063
- largest standardised coefficients: n_2q_mapped +0.181, routing_overhead_2q +0.171,
  log2_sampling_overhead +0.149, -log ESP +0.140; depth_a -0.003, depth_b -0.052
- H2 (gain >= 0.10 and depth among the largest): NOT MET. The information ESP misses
  is in the realised routing, not in idle time. A linear model is a lower bound on
  learnability; 89% of the variance remains unexplained.

## Q3. Budget-aware selection, geometric mean of MSE / oracle MSE (oracle = best candidate)
budget   min_overhead  max_esp  esp_budget  esp_budget within 10%
1e3      1.001         3.411    1.001       1.000
1e4      1.005         2.817    1.008       0.988
1e5      1.023         1.906    1.031       0.895
1e6      1.073         1.308    1.087       0.698
1e7      1.145         1.148    1.135       0.537
1e8      1.184         1.144    1.145       0.519
1e9      1.190         1.144    1.144       0.512
- H3: MET.
- Budget-blind selection by ESP is 3.4x worse than the oracle at 1e3 shots.
- Decision headroom within the candidate set: about 2% up to 1e5, about 14% at >= 1e7.
- At 1e6 ESP-budget is worse than min-overhead: ESP magnitudes are miscalibrated.

## Decision (pre-registered rules)
Neither "little headroom" nor "build" applies: AMBIGUOUS, discuss before building.

## Implication for the comparison with find_cuts
Within our candidates, min-overhead is within 0.1% of the best at 1e3; in the 23 Oct
comparison it was 1.90 against a best that included the find_cuts cut, and find_cuts
was 1.09. At small budgets the deficit comes from the candidate set, not the ranking.
