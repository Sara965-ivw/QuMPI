# Results: paper_budget/analysis/per_observable.py on holdout3 (scheme v3.1), 5 Nov 2026

Exploratory; holdout3 is not a test set for the paper. Predictions committed beforehand.
162 circuits, 2105 informative (circuit, observable) pairs (|mu*| > 0.2). Descriptive only.

Ratio = MSE of a rule's cut for one observable / MSE of the best cut for that observable.
budget  basket-best  share>1.10  circuits>1.10  min-overhead  share>1.10
1e3     1.010        0.030       0.000          1.011         0.034
1e4     1.058        0.157       0.167          1.066         0.181
1e5     1.195        0.393       0.556          1.221         0.422
1e6     1.361        0.561       0.864          1.436         0.607
1e7     1.463        0.616       0.889          1.604         0.676
1e8     1.499        0.617       0.901          1.677         0.694
1e9     1.507        0.615       0.901          1.694         0.695

- O1 (<= 1.03 and share <= 0.10 up to 1e5): NOT MET (holds at 1e3 only).
- O2 (<= 1.10 at 1e7-1e9): NOT MET (1.46-1.51).
- O3 (per-observable headroom over minimum overhead at 1e9 > 1.19): MET (1.694).
- Decision (pre-registered rule): an observable-conditioned output is worth building.
- Per-observable headroom over minimum overhead (1.22 at 1e5, 1.69 at 1e9) is far larger than
  the basket-level headroom (about 1.02 at 1e5, 1.19 at 1e9), and it starts at practical budgets.

Hypothesis (not yet tested): much of the gap comes from readout placement. Each cut maps the
measured logical qubits to physical qubits with different readout errors; with symmetric
readout, <Z_S> is scaled by prod_{q in S}(1 - 2 e_phys(q)). Readout error mitigation would
remove this part, and an observable-level ESP would capture it without learning.
Before building: (1) remove the readout factor exactly and recompute the per-observable
headroom; (2) add an observable-level ESP baseline.
