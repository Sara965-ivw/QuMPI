# Predictions for the Qdislib comparison, recorded BEFORE the full Qdislib run (5 Nov 2026)

Data: the 143 holdout3 circuits of the find_cuts comparison (exploratory; confirmation on holdout4).
"Qdislib" = default find_cut at the widest width that fits after routing, averaged over 5 seeds.

Structural (checkable from the cuts, before any labelling):
S1. On the separable circuits, where find_cuts needs no cut, Qdislib returns a cut in most of them.
S2. On connected circuits, Qdislib's sampling overhead gamma^2 is >= find_cuts' in at least 90%
    (find_cuts minimises gamma; Qdislib minimises a cut-count/size loss).
S3. SparseCut has no feasible cut (max_cuts <= 6, fragments <= 13 routed qubits) in most circuits:
    it targets routing distance, not fragment width.

After labelling (same labeller as every other cut):
L1. At 1e3 shots, find_cuts has lower MSE than Qdislib in at least 70% of circuits.
L2. Qdislib's Hellinger fidelity does not differ from random selection (noise-blind, like find_cuts).
L3. Adding Qdislib's cuts (default, top5, mixed) to the candidate set lowers the in-set oracle MSE
    at 1e3 in fewer circuits than adding the find_cuts cut does.
