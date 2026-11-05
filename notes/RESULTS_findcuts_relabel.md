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
