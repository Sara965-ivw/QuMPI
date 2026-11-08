# Amendment 1 to PREREG_holdout4.md: M2b by overhead group

Committed while holdout4 is being generated and before any holdout4 label or result has been read.
Nothing in PREREG_holdout4.md changes: H5 stays the primary hypothesis, the models stay frozen
(same 15 SHA-256), and eval_holdout4.py is not modified.

## Origin (disclosed)
Found post hoc on holdout3 (m2b_by_overhead.py, RESULTS_m2b_holdout3.md): M2b's whole gain over M1 came
from the 39 of 162 circuits whose candidate cuts all share one sampling overhead (M2b/M1 = 0.902 at 1e8,
0.901 at 1e9; 28 of 39 circuits with different choices). On the 123 circuits with several overheads the
ratio was 1.000 and 0.998. Mechanism proposed: with equal overheads the choice rests on the noise ranking
alone, which is what M2b improves (Kendall tau 0.338 vs 0.252).

## Definitions
Analysis set: as in PREREG_holdout4.md section 3. A circuit is "one overhead" when all its candidates
have the same sampling overhead (log2 kappa^2 equal after rounding to 1e-9); otherwise "several".

## Hypotheses (own family, Holm over 2; run by eval_holdout4_h7.py, committed with this file)
- H7: on one-overhead circuits, M2b's geometric-mean regret is at least 5% lower than M1's
  (ratio <= 0.95) at 1e8 AND 1e9, with Holm-adjusted p < 0.05 at both. Not evaluated (reported as such)
  if fewer than 5 such circuits.
- H8 (descriptive): on circuits with several overheads, the M2b/M1 ratio is within 2% of 1 at 1e8 and 1e9.
- Reported: the share of one-overhead circuits in holdout4 (holdout3: 24%).

## Why these thresholds
The holdout3 effect in the group was about 10%; the margin is half of it. With about 40 circuits the
power is uncertain; the CI is reported either way.

## Decision
- H7 met: "the Graph Transformer improves cut selection among equal-overhead candidates" is claimed, with
  the origin of the hypothesis stated.
- H7 not met: the group finding stays an exploratory holdout3 observation.
- H5 and H7 are reported side by side; neither rescues the other.
