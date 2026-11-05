# Predictions: budget-aware selection without the ideal values (deployable_rule.py)

Written and committed before running. Data: holdout3 (already unblinded, exploratory).

## Question
headroom.py Q3 and learnability.py A2 chose a cut by b2_hat + v_hat / N with mu_hat = f_hat * mu*.
They used the ideal values mu*, which a user does not have. Does budget-aware selection keep
its advantage when mu* is replaced by something a user can obtain?

With one attenuation f_k per cut, the rule depends on mu* only through m = mean_S mu*_S^2:
b2_hat_k = (1 - f_k)^2 m and v_hat_k = gamma2_k - f_k^2 m (no clipping, since gamma2 >= 9 > mu^2).

## Fixed design (no tuning after seeing results)
- f_hat: out-of-fold linear ESP fit, same 2 folds and SPLIT_SEED as headroom.py.
- Rules: min_overhead, max_esp, obs_true (Q3 rule), m_true, m_one (m = 1), m_x0.25, m_x4,
  m_pilot (10% of shots on the minimum-overhead cut, m estimated with the unbiased square
  correction, clipped to [0, 1]; the other 90% on the chosen cut; 20 simulated pilots per
  circuit and budget, Gaussian approximation, seeded), ceiling_true_att (true attenuation and mu*).
- Metric: geometric mean over circuits of MSE / oracle MSE at the full budget N.
- Tests: paired per circuit, log ratio, Wilcoxon, Holm over the 7 budgets.

## Checks (must hold, otherwise stop and inspect)
- C1. obs_true reproduces the Q3 'esp_budget' column of notes/RESULTS_headroom.md to 3 decimals.
- C2. obs_true and m_true choose different cuts in at most 1% of (circuit, budget) cases.

## Predictions
- P1. m_one is more than 5% worse than m_true (geometric-mean ratio) at one or more budgets in
  1e4-1e6, and within 2% of it at 1e8 and 1e9 (where every rule picks the lowest-bias cut).
- P2. m_x0.25 and m_x4 are within 10% of m_true at every budget (the rule is robust to a
  factor-4 error in m).
- P3. m_pilot / m_true is between 1.05 and 1.12 at 1e3 (cost of the pilot, about 1/0.9),
  and at most 1.05 at every budget >= 1e6.
- P4. m_pilot beats min_overhead at every budget >= 1e6 and beats max_esp at every budget
  <= 1e5 (Holm-adjusted p < 0.05 in each case).

## Decision rule
- P3 and P4 hold: the budget-aware rule is usable without mu*; continue with the roadmap
  (candidate union, then the attenuation model), keeping the pilot as part of the method.
- P3 fails at >= 1e6 or P4 fails: estimating m becomes its own component (reuse pilot shots,
  estimate m from an uncut noisy run, or choose the rule that is safe over a range of m)
  before any further model work.
- C1 or C2 fails: stop; the script or the data differs from headroom.py.

## Limitations stated in advance
- Gaussian pilot approximation; exact for large pilots, rough at 100 shots (N = 1e3).
- Pilot shots are discarded (conservative); reusing them would lower m_pilot's cost.
- One attenuation per cut; per-observable attenuation (per_observable.py) is not modelled.
- Candidates only (no find_cuts or Qdislib cuts): the union is the next step.
