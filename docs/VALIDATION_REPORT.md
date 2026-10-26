# QuMPI release-candidate validation report

Validation was performed on the release-candidate copy, not the original WSL working directory. No model training, corpus generation, holdout evaluation, or MPI simulation was run during this validation.

## Passed checks

- The supplied research handoff archive's SHA-256 matches the collection report.
- All 792 file entries in the handoff archive's `SHA256SUMS.txt` verified successfully.
- All 10 `.pt` model checkpoint files in the handoff match the hashes recorded in `partB_record.json`.
- All 672 bundled corpus files are covered by the frozen artifact manifest; the checksum manifest verifies all 776 frozen data/result/record artifacts in this candidate.
- The original code files copied from the supplied source snapshot match the source snapshot SHA-256 values; `.gitignore` is the only inherited root configuration file intentionally revised.
- `scripts/verify_artifacts.py` ran successfully and reported 776 matching artifacts.
- `scripts/compare_rq1.py` passed a self-consistency check on the frozen per-circuit CSV (1,430 selected primary/GBM rows); this validates the comparator's row-selection/path logic only, not a fresh RQ1 reproduction.
- `bash -n scripts/quest.sh` passed.
- Re-analysis of the existing `runs/rq2_prelim/runs.jsonl` in a temporary directory reproduced the archived `per_run.csv` and `summary.txt` byte-for-byte. No MPI job was launched. The newly rendered PNG was not required to match byte-for-byte.
- A bounded text scan found no obvious private-key/API-token patterns or raw user-home paths in the candidate text files.
- Generated Python bytecode, `__pycache__`, `.pytest_cache`, `.git`, and environment directories are absent from the candidate.

## Not performed / still required before public release

- No fresh RQ1 evaluation was executed; the wrapper is prepared but requires explicit `--confirm` because it re-reads the already-unblinded holdout3.
- No MPI pilot was re-run. The known dynamic shared-counter contention and cost-model calibration issues remain open.
- No full test suite or clean-machine dependency installation has been run.
- The live WSL repository state and corpus were not accessed directly by this validation process; the user's earlier terminal reported clean `master` at `c076020`, and the collection record documents a corpus hash check at 14:51 on 25 October 2026.
- No repository was initialized, committed, pushed, or published.
- A repository licence and institutional/public-release permission have not been selected; see `PUBLIC_RELEASE_CHECKLIST.md`.

## Collection-record discrepancy

The original handoff `README_MANIFEST.txt` says 789 files were copied, while its `inventory.csv` contains 790 `copied` rows and the extracted package contains 790 non-metadata artifact files. All checksum entries pass. This appears to be a one-file documentation/count discrepancy, not a content-hash mismatch.
