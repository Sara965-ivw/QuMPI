# Results provenance

## Source snapshot

- Repository reported by the user: `Sara965-ivw/quest-mpi`
- Branch / commit reported by the user: `master`, `c076020bf4017158b4a72e9cc72002dfe0617e84`
- Source archive supplied: `QuMPI_source_snapshot.tar.gz` (the archive omitted `.git`, caches were excluded from this candidate). The claim that the live working tree was clean comes from the user's terminal output; the snapshot itself cannot independently prove Git status because `.git` is absent.

## Primary RQ1

- Original frozen outputs: `runs/rq1_h3/`
- Registered analysis set: `data/corpus_v2/holdout3_selection.json` (143 selected circuits)
- Checkpoint record: `runs/rq1_h3/partB_record.json` and duplicate root record `runs_records/partB_record.json`
- Recorded main outputs: `per_circuit.csv`, `summary.csv`, `summary_by_family.csv`, `summary_by_width.csv`, `holdout3_tests.json`, `log.txt`
- Model/GBM comparison reproduction log: included in the collection archive, not copied into this candidate because it includes raw collection-specific path context; the run result files and hashes are retained.

## RQ2

- Original timing records: `runs/rq2_prelim/runs.jsonl`
- Original analysis: `runs/rq2_prelim/summary.txt`, `per_run.csv`, `rq2_prelim.png`
- The raw pilot is retained unchanged. Its records identify the run environment and commit (`ccf179e`) rather than the current repository head.

## Integrity manifests

- `data_manifest/FROZEN_ARTIFACTS_SHA256SUMS.txt` uses the source hashes recorded in the verified handoff inventory for the data and run artifacts.
- `data_manifest/SOURCE_SHA256SUMS.txt` records the SHA-256 of original source files copied without modification from the supplied code snapshot, except `.gitignore`, whose release-candidate rules were intentionally revised and whose previous text is retained in `docs/ORIGINAL_GITIGNORE.txt`.

Packaging checks verify files and hashes only; they do not reproduce scientific results.
