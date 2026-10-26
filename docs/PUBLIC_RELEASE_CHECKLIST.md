# Public release checklist (must be completed before pushing)

- [ ] Confirm King Khalid University / supervisor / institutional IP and release permission.
- [ ] Choose and obtain approval for a repository licence. No licence is asserted in this candidate.
- [ ] Confirm whether corpus, already-unblinded holdout3 data, all checkpoint files, and historical logs may be public now or only after paper submission.
- [ ] Review third-party dataset, IBM calibration fixture, Qiskit and dependency licensing/attribution terms.
- [ ] Decide whether public Git should contain all 672 corpus files / weights / run outputs or link to a versioned archive/DOI; keep the checksums either way.
- [ ] On a clean machine, install a pinned environment and run unit tests. Record exact OS, Python, CPU/GPU, CUDA, MPI and package versions.
- [ ] Execute read-only artifact verification from a fresh clone.
- [ ] Run the RQ1 reproduction wrapper only with explicit opt-in, compare outputs against frozen result tables, and document non-bit-reproducible components (notably random baseline behavior).
- [ ] Re-run RQ2 **analysis only** from the existing JSONL and compare to the archived summary. Do not rerun MPI timing until the shared-counter and cost model issues are fixed.
- [ ] Correct the one-file total mismatch in the original handoff README in the collection record if the handoff archive is redistributed.
- [ ] Confirm final authorship, affiliation, citation/BibTeX, version/tag, DOI plan and publication timing.
- [ ] Do not label RQ2 pilot as the completed RQ2 study; do not claim Stage 3 QUBO or RQ3 are implemented.
