# Predictions recorded BEFORE running per_observable.py (exploratory, holdout3 v3.1)

Same circuits as headroom.py (>= 3 candidates); observables with |mu*| > 0.2 only.
Ratio = MSE of a rule's cut for one observable / MSE of the best cut for that observable.

O1. Up to 1e5 shots the basket-best cut is also (near) best per observable: geometric-mean
    ratio <= 1.03 and share of pairs above 1.10 <= 0.10 (variance dominates and kappa^2 is
    shared by all observables of a cut).
O2. At 1e7-1e9 shots the basket-best cut stays close: geometric-mean ratio <= 1.10
    (the light-cone analysis suggested that cut effects are mostly global, through routing).
O3. Per observable, the headroom over minimum overhead at 1e9 is larger than the basket-level
    value (1.19).

Decision rule:
- O2 fails (ratio > 1.10 at some budget >= 1e7) or more than 25% of circuits exceed 1.10:
  an observable-conditioned output is worth building.
- Otherwise the basket-level target is sufficient; observable conditioning is not pursued.
