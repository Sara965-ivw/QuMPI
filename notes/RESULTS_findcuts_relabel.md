# Results: regenerated find_cuts labels (5 Nov 2026)

Unchanged script analysis/findcuts_label.py, seed 1234, width 13, 1800 s limit, 4 workers; 35 min.
- R1 (reproducibility): NOT FULLY MET. 142 of 143 circuits match October exactly (status, width
  used, cut type, gamma^2, label fidelity to 9 decimals). One circuit differs:
  holdout3_clustered_random_n18_s97185: same status, width, cut type and gamma^2, but label
  fidelity 0.960175744 (October) vs 0.958552325 (now). Cause under investigation; the labels
  are not used until it is explained.
  Width distribution identical: 13: 82, 12: 22, 11: 32, 10: 4, 9: 2.
- R2: holdout3_clustered_random_n20_s97365 exceeded the time limit again: MET.
(Correction: an earlier commit of this file stated R1 as met in every circuit, before the
 per-circuit check had been read.)

## Explained (5 Nov 2026, before any label was used)
- find_cuts is deterministic: the same cut (gate 69, width 10) under PYTHONHASHSEED 0-4.
- Both October and tonight labelled that cut afresh (no reused candidate). The multipiece
  labeller is not deterministic across Python hash seeds: the fourth piece (logical qubits
  8, 13, 14) is placed on the same physical qubits {112, 125, 126} in a different order.
  PYTHONHASHSEED=0 and =1 give [125, 112, 126] and F = 0.960175744 (October exactly);
  tonight gave [112, 125, 126] and F = 0.958552325. A tie between equal-scoring layouts is
  broken by set iteration order, which the transpiler seed (1234) does not control.
- Effect: 1 of 143 circuits, |dF| = 0.0016. This is a label noise floor to be quantified.
