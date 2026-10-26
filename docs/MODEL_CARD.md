# Model card: `gt_size-layout`

## Purpose
Rank candidate cuts for a single circuit using graph/candidate features and calibration/layout information from the recorded IBM `FakeBrisbane` backend snapshot. The intended output is an ordering of candidate cuts, not an absolute calibrated fidelity estimate.

## Frozen model record

- Primary: `gt_size-layout`; control: `gt_size-layout_uniform`.
- Architecture in the recorded run metadata: Graph Transformer, hidden dimension 64, 4 attention heads, 3 layers, `k_pe=8`, dropout 0.1, additive features, size-free scoring, random-walk positional encoding.
- Training circuits: nominal width 6–14 qubits; corpus generation uses three families (clustered random, QAOA on community/SBM graphs, and two-local ansatz).
- Frozen seeds: 0–4. Split seed: 2026. Transpiler seed: 1234.
- Training configuration recorded in `runs_records/partB_record.json`; SHA-256 checkpoint hashes are also checked by `scripts/verify_artifacts.py`.
- Confirmatory evaluation: 143 selected circuits at 16, 18, and 20 qubits.

## Performance interpretation

The primary model's recorded gap-weighted Kendall score is approximately 0.404. It outperforms random, minimum-overhead, and METIS in the registered comparisons. Its difference from GBM-layout and layout-ESP is not statistically significant under the recorded tests. Do not claim that the Graph Transformer architecture itself is necessary or uniquely superior.

## Limitations

- One calibration snapshot / backend model; no real quantum-hardware execution.
- Tested generalisation only to widths 16–20; no result here validates the proposed 22–40 qubit range.
- Training on GPU was reported not to be bit-for-bit deterministic; inference was reported to reproduce the original 1,430 comparison values under the documented current code.
- Training labels require fragments to fit the local simulation limit; this restricts which circuits and cuts could be labelled.
- Infinite-shot noisy fidelity rankings do not necessarily identify the best finite-shot choice; consult the finite-shot budget analysis.
