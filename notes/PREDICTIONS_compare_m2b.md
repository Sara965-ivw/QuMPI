# Predictions: frozen M2b vs IBM find_cuts and Qdislib (compare_tools_m2b.py)

Written and committed before running. Exploratory (holdout3; PREREG_holdout4.md section 8).
Same circuits, labels, oracle and tests as compare_tools.py; M2b and M1 frozen (SHA-256 verified),
budget-aware with the true m.

- T1. Connected circuits, 1e8 and 1e9: M2b has at least 15% lower error than find_cuts
  (geometric ratio find_cuts / M2b >= 1.15, Holm p < 0.05).
- T2. Connected circuits, 1e8 and 1e9: M2b has at least 5% lower error than Qdislib (ratio >= 1.05).
- T3. 1e3, all circuits: find_cuts and Qdislib both have lower error than M2b (ratio < 1), because our
  candidate set lacks their lowest-overhead cuts.
- T4. M2b is at least as good as M1 against both tools at every budget >= 1e6 (its ratio is not lower).
