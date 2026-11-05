# Predictions recorded BEFORE running learnability.py (exploratory, holdout3 v3.1)

Holdout3 is used for design decisions only. Same 162 circuits and attenuation as headroom.py.
Models are fitted out of fold (5 folds grouped by circuit); GBM hyperparameters fixed in advance.

P1. gbm_esp_cut reaches within-circuit R^2 >= 0.30 (nonlinear learnable signal).
P2. gbm_esp_cut_budget is <= 1.08 x oracle at 1e9 (ESP-budget was 1.144).
P3. The ceiling of a single attenuation per candidate (true_att_budget) is <= 1.03 x oracle
    at every budget.

Decision rules:
- P1 or P2 met: nonlinear signal exists; continue with B (stronger ESP), then the residual model.
- P1 and P2 both fail and R^2 < 0.15: simple features add little; prioritise C (candidate union)
  and treat the residual model as an exploratory bet.
- P3 fails (ceiling > 1.05 at some budget): one number per candidate is not enough;
  the model must predict per observable.
