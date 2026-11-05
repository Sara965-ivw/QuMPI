# Results: budget-aware selection without the ideal values (deployable_rule.py)

Run 6 Nov 2026 on holdout3 (exploratory), commit ed819b0; predictions committed before the run
(notes/PREDICTIONS_deployable_rule.md). 162 circuits, 1134 (circuit, budget) cases.
Raw output: runs/paper_budget/deployable_rule/out.txt (not committed).

## Checks
- C1. obs_true column: 1.001 1.008 1.031 1.087 1.135 1.145 1.144. Matches the Q3 esp_budget
  column of RESULTS_headroom.md exactly (min_overhead and max_esp too): met. Independent
  consistency: ceiling_true_att (1.000 1.002 1.009 1.019 1.034 1.040 1.042) reproduces
  learnability.py true_att_budget.
- C2. obs_true and m_true chose different cuts in 0 of 1134 cases: met. The rule depends on the
  ideal values only through m = mean mu*^2, as derived.

## Signal size
mean mu*^2 per circuit: median 0.078, IQR [0.037, 0.121]. The worst case m = 1 overstates it
about 13-fold for the typical circuit.

## Predictions
Ratios below are geometric means of MSE / oracle, divided by the m_true value (same circuits).

| budget | 1e3 | 1e4 | 1e5 | 1e6 | 1e7 | 1e8 | 1e9 |
|---|---|---|---|---|---|---|---|
| m_one / m_true | 1.003 | 1.061 | 1.185 | 1.153 | 1.011 | 0.999 | 1.000 |
| m_x0.25 / m_true | 1.000 | 1.000 | 0.994 | 0.993 | 1.004 | 0.999 | 1.002 |
| m_x4 / m_true | 1.003 | 1.002 | 1.028 | 1.039 | 1.008 | 0.999 | 1.000 |
| m_pilot / m_true | 1.105 | 1.083 | 1.048 | 1.022 | 1.005 | 1.001 | 1.000 |
| m_true / min_overhead | 1.000 | 1.003 | 1.008 | 1.013 | 0.991 | 0.967 | 0.961 |

- P1 (m_one > 5% worse somewhere in 1e4-1e6, within 2% at 1e8-1e9): met (up to 18.5% worse at 1e5).
- P2 (factor-4 error in m costs < 10% at every budget): met (at most 3.9%).
- P3 (pilot cost 1.05-1.12 at 1e3, <= 1.05 at >= 1e6): met (1.105; 1.022 to 1.000).
- P4 (m_pilot beats min_overhead at every budget >= 1e6, Holm p < 0.05; beats max_esp at every
  budget <= 1e5): not met. Against max_esp: met (3.09x, 2.58x, 1.76x; p 8e-3, 8e-3, 4e-2).
  Against min_overhead: m_pilot is worse at 1e6 (0.966, p 4e-16), better by 0.3% at 1e7
  (p 1e-4), 3.3% and 4.0% better at 1e8-1e9 (p 0.1 and 0.2, not significant).

Tally: 3 met, 1 not met (plus 2 checks, both met).

## Interpretation
1. Not knowing mu* is cheap. The rule needs one number, m; a factor-4 error in m costs at most
   3.9%, and the 10% pilot costs 2.2% or less from 1e6 on. Its cost at small budgets (10.5% at
   1e3, 8.3% at 1e4) is the discarded pilot shots: 1/0.9 = 1.11.
2. P4 failed for a different reason than the decision rule anticipated. m_true, which knows m
   exactly, also loses to min_overhead at 1e6 (1.013) and gains only 3-4% at 1e8-1e9. The
   bottleneck is the attenuation estimate f_hat (linear ESP), not m: with the true attenuation
   (ceiling_true_att) the same rule reaches 1.019 at 1e6 and 1.042 at 1e9.
3. m_x0.25 is slightly better than m_true at 1e5-1e6 (0.994, 0.993): shrinking toward the
   minimum-overhead cut helps when f_hat is noisy. Small and exploratory; not acted on.
4. Very small Holm p-values for differences under 0.5% (e.g. max_esp / m_pilot = 0.998 at 1e8,
   p 2e-27) reflect consistent signs over 162 circuits, not meaningful effects.

## Deviation from the pre-registered decision rule (flagged)
The registered rule says: P4 fails -> make estimating m its own component before further model
work. Interpretation 2 shows that P4 failed because of f_hat, which m estimation cannot fix
(m_true fails the same comparison at 1e6). Proposed instead, pending Sara's and the
supervisor's agreement:
- Keep m handling simple: the budget-aware rule is usable without mu*.
- Remove the pilot cost by design for holdout4 (stated as a limitation before this run):
  (a) reuse the pilot shots when the chosen cut is the pilot cut, or
  (b) a fixed m from training circuits (no pilot); P2 suggests this costs a few percent at most.
  Both to be compared on the training set or by cross-fitting, not chosen on holdout3.
- Continue the roadmap with the attenuation model (step 2) and the candidate union (step 1);
  they target the measured bottleneck.

## Limitations
Gaussian pilot approximation (rough at 100 shots); one attenuation per cut; candidates only;
holdout3 has been used for several design analyses, so all of this is exploratory.

## Thesis / paper use
Methods: the derivation that the rule needs only m (C2 confirms it numerically). Results or
appendix: the table above as the answer to "the selector uses the ideal values".
