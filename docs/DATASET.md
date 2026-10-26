# Dataset card: `corpus_v2`

## Included material

This candidate bundles the collected `corpus_v2` directory byte-for-byte, including split JSON files, audit files, manifests, and `holdout3_selection.json`. The bundle inventory recorded 672 files under this directory. The files are integrity-checked using `data_manifest/FROZEN_ARTIFACTS_SHA256SUMS.txt`.

Recorded split inventory (file counts include audit/empty-label records where applicable):

| Component | Files | Role / widths (recorded) |
|---|---:|---|
| `train/` | 210 | Training pool, widths 6–14 |
| `holdout/` | 66 | Earlier, development-seen holdout |
| `holdout2/` | 45 | Earlier, development-seen holdout |
| `holdout3/` | 287 | Pre-registered, already-unblinded evaluation data plus associated audit files |
| `probe/` | 30 | Development probe |
| `probe_v3/` | 29 | Development probe |
| four manifests | 4 | Provenance/version metadata per split |
| `holdout3_selection.json` | 1 | The fixed 143-circuit analysis selection |

Counts above follow the archived inventory and are not a claim that every JSON is an independent labelled circuit: some `*.g1_empty.json` files are audit records.

## Generation and labels

The recorded RQ1 workflow uses deterministic noise-aware labels from circuit-cut fragment reconstruction and a calibration-derived model associated with `FakeBrisbane`. Primary labels are based on Hellinger fidelity of the reconstructed distribution against the ideal distribution. A noiseless oracle is retained for correctness checks. The main RQ1 test is ranking quality across candidate cuts.

## Splits and leakage cautions

- The 12–14 qubit selection subset was used for multiple model decisions; it is not an untouched final test set.
- `holdout` and `holdout2` were used during development and are not confirmatory.
- `holdout3` is already unblinded; future reruns are reproductions, not a fresh confirmatory evaluation.
- The primary confirmatory selection contains 143 circuits at 16–20 qubits. The archived report notes that 33 test circuits had no cut because the generator did not enforce connectedness; report connected-only results separately when scientifically relevant.

## Distribution and privacy

These are generated research artifacts. Before a public release, confirm institutional permission, licensing, paper timing, and whether bundled calibration/third-party materials may be redistributed. No private credentials are intended to be present.
