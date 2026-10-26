# Reproducing the released results

This guide separates integrity verification, re-analysis of existing records, and a new evaluation pass. The bundled holdout3 was already unblinded on 16 October 2026; running it again is a reproduction check, not a new confirmatory experiment.

## 1. Verify artifacts (read-only)

From the repository root:

```bash
./scripts/quest.sh verify-artifacts
```

This checks checksums for all frozen data, checkpoint, and result artifacts. It does not load circuit JSON, train models, simulate circuits, or open holdout records for scientific analysis.

## 2. Reanalyse the captured RQ2 pilot (no MPI jobs)

```bash
./scripts/quest.sh analyze-rq2
```

The script reads `runs/rq2_prelim/runs.jsonl`, writes to a new timestamped directory, and compares the regenerated `per_run.csv` and `summary.txt` against the archived analysis. It does not invoke `mpirun` or `rq2_prelim.py`.

## 3. Reproduce the recorded RQ1 evaluation (explicit opt-in)

Only do this when you intentionally want to rerun the already-unblinded evaluation and the recorded environment is installed:

```bash
./scripts/quest.sh reproduce-rq1 --confirm
```

Safety properties of the wrapper:

- Requires the explicit `--confirm` argument.
- Verifies the artifact checksum manifest first.
- Copies the ten frozen checkpoints and `partB_record.json` into a new timestamped output directory.
- Calls `run_rq1_h3.py --stage evaluate`, never `--stage freeze`.
- Does not write into or delete `runs/rq1_h3/`.
- Compares the frozen and reproduced `kendall_w` values for 715 primary-model rows and 715 GBM-layout rows; any difference fails the comparison.
- Does not commit, push, tag, install packages, or remove output on failure.

The evaluation reads the already-unblinded `holdout3/` split and its fixed `holdout3_selection.json`. Random baseline scores may not be bit-reproducible because the historical baseline uses Python's salted `hash()`; the wrapper compares the frozen primary model and GBM-layout, the recorded 1,430-value comparison noted in provenance.

## 4. Unit tests and clean installation

A clean installation and the full test suite were not run during packaging. Consult `environment/ENVIRONMENT.md`. Do not run all tests blindly on a machine with limited resources; review the test files and install the matching pinned environment first. The main artifact verification command is independent of Python ML/MPI dependencies beyond the Python standard library.
