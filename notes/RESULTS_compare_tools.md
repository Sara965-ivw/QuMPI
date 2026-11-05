# Results: compare_tools.py, our rules vs find_cuts vs Qdislib (holdout3, exploratory, 5 Nov 2026)

142 circuits (s97365 excluded as in October); Qdislib missing 0; optimistic bound used 0.
Same exact finite-shot MSE for every cut; oracle = best of candidates, find_cuts and Qdislib cuts.
Full output: runs/paper_budget/compare_stdout.txt.

Connected circuits (109), geometric mean MSE / oracle:
method        1e3   1e5   1e7   1e9   F
model         5.37  2.30  1.18  1.14  0.9249
gbm-layout    2.89  1.66  1.12  1.12  0.9252
min_overhead  1.16  1.06  1.11  1.17  0.9235
find_cuts     1.09  1.19  1.27  1.32  0.9207
qdislib       1.14  1.13  1.13  1.17  0.9233

Paired (reference/method, > 1 = method better; Holm over 7 budgets), connected:
- model vs find_cuts: 0.20, 0.52 at 1e3, 1e5 (worse); 1.08, 1.15 at 1e7, 1e9 (better, p <= 5e-3).
- min_overhead vs find_cuts: 1.11-1.16 from 1e5 to 1e9 (better, p <= 1e-3).
- model vs qdislib: worse up to 1e6; no significant difference at >= 1e7.
- find_cuts vs qdislib: Qdislib better by 10-12% at >= 1e6 (p <= 2e-5).

Predictions (notes/PREDICTIONS_qdislib.md):
- L1 (find_cuts lower MSE than Qdislib at 1e3 in >= 70%): NOT MET, 0.275.
- L2 (Qdislib fidelity = random): NOT MET; Qdislib is higher by +0.0036 (p 4.6e-7).
- L3 (Qdislib cuts lower the candidate oracle less often than find_cuts at 1e3): MET, 0.254 vs
  0.289 (deviation: Qdislib top5/mixed cuts were not labelled; default and SparseCut only).

Union headroom (candidate oracle / union oracle): all 1.90, 1.46, 1.15 at 1e3-1e5, ~1.01 above;
connected 1.16, 1.11, 1.04, ~1.00 above.

Interpretation: no tool dominates. Qdislib is the strongest baseline: same sampling overhead
as find_cuts in 97% of pairs but cleaner cuts (hypothesis: smaller, balanced fragments need
less routing). The frozen model beats find_cuts only at >= 1e7 shots and does not beat Qdislib.
Next: budget-aware selection over the union with simple bias estimates, before any training.
