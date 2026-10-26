"""QUEST-MPI Phase 1: training-data generation core (amended D1/D2/D3).

Validated in a dependency-free numpy sandbox; qiskit/Aer production path in
qiskit_bridge.py must pass the same test suite on the pinned environment
before any dataset generation (pilot exit criterion).
"""
from .candidates import CutCandidate, enumerate_candidates, validate_candidate
from .circuits import Circuit, Gate
from .combiner import label_candidate, reconstruct_noiseless, reconstruct_noisy
from .generators import qaoa_maxcut_circuit, random_circuit, twolocal_circuit
from .labeling import additivity_probe, label_circuit, run_pilot
from .noise import build_noise_spec, gate_node_features, load_snapshot
from .simulators import (NoiseSpec, QubitNoise, hellinger_fidelity,
                         probabilities, total_variation)
