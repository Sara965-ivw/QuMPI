"""Cross-validation: numpy core vs qiskit/Aer engine (pilot exit criterion).

Requires qiskit + qiskit-aer, so it runs on the pinned production environment,
not the dependency-free sandbox. All tests must be green BEFORE any dataset
generation with the Aer engine, and again after the qiskit-aer-gpu swap
(set QUEST_DEVICE=GPU to exercise the GPU path).

Test ladder (each isolates one layer, so a failure localizes the bug):
  1. Round-trip circuit conversion (gate lists identical).
  2. Noiseless engine equivalence (Aer DM diagonal == numpy probabilities).
  3. Noisy engine equivalence (full density matrices match elementwise) —
     catches endianness, noise-composition order, and depolarizing-parameter
     convention mismatches; the fixture's asymmetric T1/T2 makes qubit-order
     swaps visible.
  4. Label equivalence (end-to-end reconstruct_noisy with each engine).
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quest_phase1.candidates import enumerate_candidates
from quest_phase1.circuits import Circuit
from quest_phase1.combiner import reconstruct_noisy
from quest_phase1.generators import (qaoa_maxcut_circuit, random_circuit,
                                     twolocal_circuit)
from quest_phase1.noise import build_noise_spec, load_snapshot
from quest_phase1.qiskit_bridge import (aer_density_matrix, from_qiskit,
                                        to_qiskit)
from quest_phase1.simulators import (density_matrix, dm_diagonal,
                                     hellinger_fidelity, probabilities)

FIXTURE = Path(__file__).resolve().parents[1] / "quest_phase1" / "snapshot_fixture.json"
DEVICE = os.environ.get("QUEST_DEVICE", "CPU")


def aer_dm_func(circ, noise=None, init_psi=None):
    return aer_density_matrix(circ, noise, init_psi=init_psi, device=DEVICE)


def suite_circuits():
    return [random_circuit(4, 3, 0.8, seed=21)[0],
            random_circuit(5, 2, 0.6, seed=22)[0],
            twolocal_circuit(5, 2, seed=23)[0],
            qaoa_maxcut_circuit(5, 1, 0.5, seed=24)[0]]


class TestConversion(unittest.TestCase):
    def test_roundtrip(self):
        for circ in suite_circuits():
            back = from_qiskit(to_qiskit(circ))
            self.assertEqual(
                [(g.name, g.qubits) for g in circ.gates],
                [(g.name, g.qubits) for g in back.gates])
            for g1, g2 in zip(circ.gates, back.gates):
                np.testing.assert_allclose(g1.params, g2.params, atol=1e-12)


class TestEngineEquivalence(unittest.TestCase):
    def test_noiseless_distributions(self):
        for circ in suite_circuits():
            p_np = probabilities(circ)
            rho = aer_density_matrix(circ, None, device=DEVICE)
            p_aer = dm_diagonal(rho, circ.n_qubits)
            np.testing.assert_allclose(p_aer, p_np, atol=1e-8,
                                       err_msg="noiseless mismatch (endianness?)")

    def test_noisy_density_matrices(self):
        snap = load_snapshot(FIXTURE)
        for circ in suite_circuits():
            noise = build_noise_spec(snap, circ.n_qubits)
            rho_np = density_matrix(circ, noise=noise)
            rho_aer = aer_density_matrix(circ, noise, device=DEVICE)
            diff = float(np.max(np.abs(rho_np - rho_aer)))
            self.assertLess(diff, 1e-6,
                            f"noisy DM mismatch {diff:.2e} — check noise "
                            f"composition order / expand-vs-tensor / depol convention")


class TestLabelEquivalence(unittest.TestCase):
    def test_labels_match_between_engines(self):
        snap = load_snapshot(FIXTURE)
        checked = 0
        for circ in suite_circuits():
            noise = build_noise_spec(snap, circ.n_qubits)
            p_ideal = probabilities(circ)
            for cand in enumerate_candidates(circ, k_max=2)[:3]:
                p_np = reconstruct_noisy(circ, cand, noise)
                p_aer = reconstruct_noisy(circ, cand, noise, dm_func=aer_dm_func)
                f_np = hellinger_fidelity(p_np, p_ideal)
                f_aer = hellinger_fidelity(p_aer, p_ideal)
                self.assertLess(abs(f_np - f_aer), 1e-6,
                                f"label mismatch: numpy {f_np} vs aer {f_aer}")
                checked += 1
        self.assertGreater(checked, 6)


if __name__ == "__main__":
    print(f"Cross-validating numpy core vs Aer (device={DEVICE})")
    unittest.main(verbosity=2)
