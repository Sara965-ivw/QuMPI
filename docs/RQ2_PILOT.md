# RQ2 pilot: results and scope

## Recorded workload

`runs/rq2_prelim/runs.jsonl` contains 576 timed records: 24 workloads × {4, 8} ranks × 4 policies × 3 repeats. Workloads are drawn from the **training** split, not holdout3. The policies are fragment grouping, equal task count (`count`), static longest-processing-time-first (`lpt`), and dynamic self-scheduling.

## Recorded metric

The analysis defines per-rank barrier wait as `elapsed - own task time`, and `rho_max` as the maximum wait-to-busy ratio among active ranks; inversion is `rho_max > 1`. `runs/rq2_prelim/summary.txt` contains the original summary and intervals.

## Interpretation restrictions

This remains a pilot, not the completed factorial experiment described in the thesis proposal. The recorded thesis discussion identifies two implementation concerns: (1) dynamic scheduling uses a shared counter on rank 0 while rank 0 also performs work, creating contention; (2) the predicted cost model did not track wall time with the assumed scaling because fixed task overhead dominated some runs. The historical policy ranking should not be presented as a validated, general scheduler comparison. The `count` policy is equal task count, not an invocation of METIS.

The current RQ2 code's usage paths are explicit command-line arguments. Running `rq2_analyse.py` on the existing `runs.jsonl` is post-processing only; running `rq2_prelim.py` executes new MPI simulations and should not be done as part of read-only validation.
