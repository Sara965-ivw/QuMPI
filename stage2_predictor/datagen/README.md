# QUEST-MPI Phase 1 — Training-Data Generation Core

Implements the amended Phase 1 kickoff decisions (see
`QUEST-MPI_Phase1_Kickoff_Decision_Record.docx`, incl. the D1 amendment of
2026-07-18): **deterministic noisy-simulation PST labels** — calibration-derived
noise model, density-matrix fragments, infinite-shot limit — over wire-cut
candidates enumerated as ancestor-closed bipartitions.

## Why two simulation stacks

The scientific core (QPD contraction, frame decomposition, candidate
enumeration, noise physics) is implemented dependency-free in numpy and
validated at machine precision in `tests/test_phase1.py` (15 tests, all green,
2026-07-18). The production path (`quest_phase1/qiskit_bridge.py`) swaps in
qiskit/Aer for GPU-scale simulation and live IBM calibration; it is **not yet
validated** — running the suite plus the bridge cross-check on the pinned
environment is a pilot exit criterion.

## Key invariants (enforced by tests, in order of importance)

1. **Oracle exactness** — noiseless QPD reconstruction ≡ ideal distribution to
   1e-10 on every candidate. The exactness of the QPD identity makes this the
   correctness oracle for the contraction; it is checked on every labelled
   candidate by default (`check_oracle=True`).
2. **Zero-noise equivalence** — the noisy (density-picture, 4^k frame-basis)
   path with an empty noise model reproduces the oracle.
3. **Label informativeness** — with brisbane-class noise, labels < 1 and vary
   across cut locations (amended-D1 requirement).
4. **Noise monotonicity** — worse T1/T2 ⇒ lower mean fidelity.

## Module map

| Module | Role |
|---|---|
| `circuits.py` | Minimal circuit/DAG representation, tensor conventions |
| `simulators.py` | Statevector + density-matrix sims, noise channels (depolarizing, thermal relaxation, readout) |
| `candidates.py` | Wire-cut candidate enumeration (ancestor closure ⇒ quotient-acyclic by construction) |
| `combiner.py` | Noiseless oracle + noisy labeler (the 4^k frame-basis contraction) |
| `noise.py` | Calibration snapshot → features (both min/mean channels) + NoiseSpec |
| `generators.py` | Seeded random / TwoLocal / QAOA-MaxCut families |
| `labeling.py` | Pilot loop, label histogram (F3 check), additivity probe (F2) |
| `qiskit_bridge.py` | Production Aer/IBM-runtime path (UNVALIDATED — see header) |

## Run

```bash
python3 tests/test_phase1.py            # full suite, ~4 s, no deps beyond numpy
python3 - <<'EOF'                       # mini-pilot demo
import sys; sys.path.insert(0, '.')
from quest_phase1 import *
from quest_phase1.labeling import run_pilot
circs = [random_circuit(5, 3, 0.8, seed=i) for i in range(3)]
out = run_pilot(circs, 'quest_phase1/snapshot_fixture.json', 'pilot.json', k_max=2)
print(out['summary'])
EOF
```

Mini-pilot reference result (sandbox, 2026-07-18): 6 circuits, 70 candidates,
6.3 s CPU; labels mean 0.986, std 0.011, range [0.964, 1.000] — note the
compression toward 1 at small scale: **the F3 histogram check is empirically
live**; revisit the logit-transform decision at the real 10-circuit pilot.

## Assumptions on record (thesis Ch. 5 / Ch. 7)

- Ideal cut instrument (no noise on the virtual measure/prepare at the
  frontier); gate + readout noise inside fragments. v1 limitation; a
  frontier-noise channel is a candidate ablation.
- Depolarizing parameter taken as the calibrated gate error (average-fidelity
  correction factor deferred; documented approximation).
- Pre-mapping assumption (D3(iii)): unlisted qubit pairs get device-class
  default 2q error.
- A "candidate" is a *cut configuration* (ancestor-closed bipartition with k
  crossing wires), not a single DAG edge — see docstring in `candidates.py`;
  this wording matters in thesis text.

## Repo integration

Copy `phase1_code/` into the quest-mpi repo (e.g. as `stage2_predictor/datagen/`),
commit with the test suite green, and tag the Phase 1 kickoff. Then: qiskit-aer-gpu
install → `test_bridge.py` cross-validation (endianness warning in
`qiskit_bridge.py` header) → 10-circuit pilot → scale.
