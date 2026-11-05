# Predictions: residual attenuation model (residual_att.py)

Written and committed before running. Data: holdout3 (exploratory, already used for design).

## Question
deployable_rule.py showed the bottleneck of budget-aware selection is the attenuation estimate
f_hat (linear ESP), not m: at 1e9 the rule reaches 1.144 x oracle with linear ESP and 1.042 with
the true attenuation. Can a correction learned on top of ESP close part of that gap?
learnability.py could not tell, because it trained GBM on the raw attenuation (design flaw).

## Fixed design (no tuning after seeing results)
- att_hat = lin(e) + g(x) - mean over the circuit's candidates of g(x).
  The correction is trained on the within-circuit residual of the linear ESP fit, so with no
  signal it is zero and the model is ESP (residual / zero-initialised design).
- Models: lin_esp; lin_esp_feats (linear correction on demeaned ESP and the 7 cut features);
  gbm_resid (learnability.GBM settings, learnability.design inputs).
- GroupKFold(5) by circuit, linear fit refitted in each fold.
- Selection: argmin (1 - f)^2 m + (gamma2 - f^2 m) / N with the true m for every model
  (isolates the attenuation estimate).
- Metrics: out-of-fold within-circuit R^2 and median Spearman; geometric mean MSE / oracle;
  share of the lin_esp-to-ceiling gap closed; paired Wilcoxon on log ratios, Holm over 7 budgets.

## Predictions
- P1. lin_esp_feats within-circuit R^2 is between 0.08 and 0.15 (headroom.py Q2 gave 0.112 with 2 folds).
- P2. gbm_resid within-circuit R^2 exceeds lin_esp_feats by at least 0.03 (nonlinear signal).
- P3. gbm_resid closes at least 25% of the lin_esp-to-ceiling gap at 1e8 and at 1e9.
- P4. gbm_resid is never more than 1% worse than lin_esp (geometric mean) at any budget.

## Decision rule
- P3 met: cut and routing features carry decision-relevant signal; gbm_resid becomes the
  baseline the Graph Transformer must beat (same residual design, graph inputs).
- P3 not met: these features do not carry the signal at decision level. The Graph Transformer
  is justified only if graph structure after routing adds information; test that next, and
  report the limitation for contribution 2.

## Limitations stated in advance
Candidates only; one attenuation per cut; holdout3 reused; the GBM hyperparameters were chosen
for raw attenuation in learnability.py and are not retuned.
