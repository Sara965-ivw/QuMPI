# Prediction before regenerating the find_cuts labels (5 Nov 2026)

The October find_cuts labels are not on this machine (only runs/findcuts_compare).
They are regenerated with the unchanged script analysis/findcuts_label.py (decisions A1-A7,
fixes A-C), seed 1234, qubits_per_subcircuit 13, time limit 1800 s per circuit.

R1. Reproducibility: for every circuit, the status, width used, cut type and gamma^2 equal
    those in runs/findcuts_compare/per_circuit.csv (October).
R2. holdout3_clustered_random_n20_s97365 exceeds the time limit again.
Any difference is reported and explained before the labels are used.
