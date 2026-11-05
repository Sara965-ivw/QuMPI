# Decisions: Qdislib 1.0.1 as a baseline (5 Nov 2026)

Taken by Sara Alshehri (recommended options accepted).

- Q1 modes (a): baselines = default find_cut and SparseCut (hardware-aware);
  top_n=5 (deduplicated) and mixed feed the candidate union only.
- Q2 width (a): same rule as find_cuts (A3, A5): max_qubits 13, lowered by one while a
  fragment exceeds 13 qubits after routing. Computed up front for widths 13..5; the widest
  feasible cut is kept. SparseCut ignores max_qubits: max_cuts k = 1..6, smallest feasible k kept.
- Q3 unrepresentable cuts (a): report the count and an optimistic lower bound
  MSE >= (gamma^2 - 1) / N (zero bias); extend the labeller only if that bound beats
  the method it is compared with.
- Q4 randomness (a): seeds 1234, 2027, 3141, 4242, 5555; results averaged over seeds.
- Q5 data (a): the 143 circuits of the find_cuts comparison (holdout3_selection.json),
  exploratory; the confirmatory comparison is on holdout4.
- Evaluation: the same labeller as every other cut (scheme v3.1, fake_brisbane, transpiler
  seed 1234). Only cut selection is compared, not Qdislib's own reconstruction.
- Observed in the smoke test (5 Nov): mixed mode in 1.0.1 does not receive max_qubits;
  top_n returns duplicates; wire-only mode returned 5 wire cuts where one gate cut suffices.
  These are properties of the released tool; it is used unmodified.
- Pilot first: 3 circuits, width 13, seed 1234, SparseCut k=1, to measure call times
  before the full grid (61 calls per circuit).

## Amendment 1 (5 Nov 2026, before any Qdislib cut was selected or labelled)
- A cut with more than 6 gate cuts is not checked for routed width (6^g QPD templates;
  the October find_cuts search hung at small widths). It is recorded as too_many_cuts and
  treated like an unrepresentable cut (Q3): optimistic bound MSE >= (gamma^2 - 1) / N.
  Justification: 6 CX cuts give gamma^2 = 9^6 ~ 5.3e5; the largest find_cuts overhead on the
  same circuits was 9^5 = 59049.
- A Qdislib call that returned an error is never read as "no cut".
