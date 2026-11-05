# Predictions recorded BEFORE running headroom.py (exploratory, holdout3 v3.1)

Holdout3 is used for design decisions only; it is not a test set for the paper.
Attenuation = mean over observables with |mu*| > 0.2 of -ln(mu / mu*), per candidate.

H1. Layout-ESP explains part, but not all, of the within-circuit attenuation:
    median per-circuit Spearman(-log ESP, attenuation) > 0 and pooled R^2 in [0.2, 0.8].
H2. The remainder is partly learnable: out-of-fold R^2 gain from simple cut features >= 0.10,
    with depth_a or depth_b among the largest coefficients (ESP ignores idle decoherence).
H3. Selection by ESP with the budget is within 10% of the oracle at budgets <= 1e5
    and more than 10% away at budgets >= 1e7.

Decision rules:
- R^2 (ESP only, out-of-fold) >= 0.8 AND ESP-budget within 1.10 of the oracle at every budget:
  little headroom; the residual model is not worth building on this data.
- Gain >= 0.10 OR ESP-budget above 1.25 at some budget: build the residual model
  (ESP + learned correction initialised at zero) as part of contribution 2.
- Otherwise: ambiguous; discuss before building.
