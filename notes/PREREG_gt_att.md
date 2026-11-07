# Pre-registration: residual attenuation Transformer

Committed before the full training run (train_gt_att.py refuses to run without it) and before any
held-out evaluation. The counts below are descriptive statistics of the finished train_v31
relabelling (notes/RESULTS_relabel_train.md), not results. Plan and derivations: notes/PLAN_gt_att.md.
Objective decision: notes/DECISIONS_objective.md.

## 1. Question
Does a Graph Transformer that learns the within-circuit part of the attenuation that layout-ESP
misses (M2) choose cuts with lower error at a given shot budget than ESP with a linear correction (M1)?

## 2. Frozen design
- Training data: train_v31 (scheme v3.1, 210 circuits, 6-14 qubits). Usable for training (>= 2
  candidates, defined label, layout-ESP, label spread): 169 circuits, 3,150 candidates; split
  stratified by family and width, seed 2026, circuit level: train 134, validation 35.
  Usable circuits by family: clustered_random 67, qaoa_sbm 32, twolocal 70 (every family >= 10,
  so the family-coverage rule of section 5 is not triggered). 24 circuits have no informative
  observable and are excluded. Attenuation labels of usable circuits: min +0.0219, median +0.0924,
  max +0.4862. a_min is computed by train_gt_att.py from the 134 training circuits and recorded in
  config.json.
- Code: att_data.py, models_att.py, train_gt_att.py and eval_gt_att.py as of commit 03922a0
  (eval_gt_att.py was tested only on the smoke run's own validation circuits, --val-check).
- Models: M0 ESP prior; M1 ESP + linear correction; M2 ESP + Transformer residual (zero-initialised,
  within-circuit demeaned); M3 direct Transformer. Architecture gt_size, run_rq1.HP, no tuning.
- Training: 5 seeds (0-4) per Transformer model; loss = per-circuit MSE of the attenuation; early
  stopping on validation loss (patience 15, at most 80 epochs); M2 keeps its initial state (= M0)
  unless validation loss improves.
- Prediction: mean of the 5 seeds' predictions (primary); per-seed results reported as spread.
  Clipped to [a_min, -ln 0.01] at prediction time; number of clipped predictions reported.
- Selection: argmin_k (1 - f_k)^2 m + (kappa2_k - f_k^2 m)/N with f = exp(-a_hat) and the true m
  (primary); 10% pilot estimate of m (secondary).
- Test sets: holdout3 (exploratory; already used for design analyses), evaluation filter as in
  deployable_rule.py (>= 3 candidates, >= 1 informative observable, layout-ESP available);
  holdout4 (confirmatory; generated after this pre-registration, PYTHONHASHSEED=0, same settings
  as holdout3).

## 3. Metrics
Per test circuit, then summarised over circuits:
- RMSE, MAE (pooled and within-circuit), R^2 (pooled and within-circuit), Kendall tau_b within circuit;
- regret rho(N) = MSE of the chosen cut / MSE of the best candidate, N = 1e3 ... 1e9;
  geometric mean over circuits and share of circuits with rho > 1.10.
Paired tests: Wilcoxon signed-rank over circuits on log regret ratios (circuits where both models
choose the same cut have a zero difference and are dropped by the test; their number is reported);
95% bootstrap CI (10,000 circuit resamples, seed 2026, all circuits) for geometric-mean ratios.

## 4. Hypotheses (Holm family: the four tests of H2 and H3)
- H1 (fit, descriptive): M2 within-circuit R^2 > M1 within-circuit R^2 (bootstrap CI reported).
- H2 (primary): geometric-mean regret of M2 is at least 2% lower than M1 at 1e8 AND at 1e9, with
  Holm-adjusted p < 0.05 at both budgets.
- H3 (value of the ESP prior): M2 regret lower than M3 at 1e8 and 1e9, Holm-adjusted p < 0.05.
- H4 (no harm, descriptive): M2 geometric-mean regret never exceeds M0's by more than 1% at any budget.

## 5. Decision criteria
- "The Graph Transformer improves budget-aware cut selection" is claimed only if H2 holds on holdout3
  AND on holdout4.
- H2 fails on holdout3: M1 is reported as the best predictor; the Transformer result is reported as
  negative; holdout4 is still run for the frozen models (no redesign is evaluated on holdout4 without
  a new pre-registration).
- H1 without H2: reported as "better fit, no decision gain".
- Family coverage: a family with fewer than 10 training circuits is reported separately as
  out-of-distribution on the test sets; the primary analysis still uses all evaluable circuits.

## 6. Secondary and exploratory (no claims)
GBM residual, minimum-kappa2 rule, IBM find_cuts and Qdislib at decision level (holdout3 labels from
compare_tools); per-family results; pilot-m results; per-seed spread.

## 7. Threats to validity (stated in advance)
Small training set; training circuits 6-14 qubits vs test 16-20; QAOA circuits may lack informative
observables (coverage reported); remaining headroom about 8% at 1e9 on holdout3, so a real but small
effect may not reach significance; holdout3 reuse; training on GPU (NVIDIA GeForce RTX 5070 Laptop, driver 591.44), so runs are
not guaranteed to be bitwise identical; seed spread is reported.
