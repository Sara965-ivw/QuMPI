# Results: Qdislib cut selection, structural checks S1-S3 (holdout3, 5 Nov 2026)

Predictions: notes/PREDICTIONS_qdislib.md (committed before the full Qdislib run).
Decisions and amendments: notes/DECISIONS_qdislib.md. No label was read.
143 circuits, 5 seeds; default find_cut: 710 (circuit, seed) pairs feasible, 5 too_many_cuts,
0 errors, 0 timeouts.

- S1 (separable circuits, Qdislib returns a cut in > 50%): MET, 165 of 165 pairs (100%).
  find_cuts needs no cut on these 33 circuits (gamma^2 = 1). Our labelled candidate sets also
  contain no no-cut option, so this subset disadvantages our model in the same way.
- S2 (connected, Qdislib gamma^2 >= find_cuts' in >= 90%): MET, 0.991 of 545 pairs:
  530 equal (97.2%), 10 greater, 5 smaller; median ratio 1.
  Unexpected: the two tools reach the same sampling overhead almost always. Equal gamma^2 does
  not imply the same cut; differences between them at a fixed overhead are differences in bias.
  Consequence: L1 (find_cuts lower MSE at 1e3 in >= 70%) is now doubtful; it is kept unchanged.
- S3 (SparseCut without a feasible cut in > 50%): MET, 0.888 (82 infeasible, 45 unrepresentable,
  16 feasible). SparseCut targets routing distance, not fragment width.

Open: the 5 pairs where Qdislib's gamma^2 is smaller than find_cuts' (search limit or the
A5 width reduction?) must be explained before the labelled comparison.
