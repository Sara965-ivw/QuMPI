"""QUEST-MPI Phase 1 regression suite (stdlib unittest; no external deps).
The scientific invariants, in order of importance:
  1. ORACLE EXACTNESS — noiseless QPD reconstruction reproduces the ideal
     distribution to machine precision on every candidate of every circuit.
     If this fails, the contraction indices are wrong and nothing downstream
     can be trusted.
  2. ZERO-NOISE EQUIVALENCE — the noisy (density-picture, frame-basis) path
     with an empty noise model reproduces the oracle. Validates the 4^k
     frame decomposition independently of the physics.
  3. LABEL INFORMATIVENESS — with brisbane-class noise, labels are < 1 and
     VARY across cut locations (the amended-D1 requirement; constant labels
     would mean zero supervision signal).
  4. NOISE MONOTONICITY — degrading T1/T2 lowers mean fidelity.
Plus: candidate validity, seeded reproducibility, loader integrity, frame
transform exactness, readout channel correctness.
"""
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quest_phase1.candidates import enumerate_candidates, validate_candidate
from quest_phase1.circuits import Circuit
from quest_phase1.combiner import (_FRAME, _TINV, label_candidate,
                                   reconstruct_noiseless, reconstruct_noisy)
from quest_phase1.generators import (qaoa_maxcut_circuit, random_circuit,
                                     twolocal_circuit)
from quest_phase1.noise import (build_noise_spec, gate_node_features,
                                load_snapshot)
from quest_phase1.simulators import (NoiseSpec, QubitNoise, apply_readout,
                                     density_matrix, dm_diagonal,
                                     hellinger_fidelity, probabilities,
                                     statevector, total_variation)

FIXTURE = Path(__file__).resolve().parents[1] / "quest_phase1" / "snapshot_fixture.json"


def ghz(n):
    c = Circuit(n)
    c.add("h", 0)
    for q in range(n - 1):
        c.add("cx", (q, q + 1))
    return c


def small_suite(seed=7):
    """A structurally diverse batch of small circuits for invariant tests."""
    circs = [ghz(4), ghz(5)]
    circs += [random_circuit(n, d, 0.8, seed + i)[0]
              for i, (n, d) in enumerate([(4, 3), (5, 3), (6, 2)])]
    circs.append(twolocal_circuit(5, 2, seed)[0])
    circs.append(qaoa_maxcut_circuit(5, 1, 0.5, seed)[0])
    return circs


class TestSimulators(unittest.TestCase):
    def test_bell_state(self):
        c = Circuit(2).add("h", 0).add("cx", (0, 1))
        p = probabilities(c).reshape(-1)
        np.testing.assert_allclose(p, [0.5, 0, 0, 0.5], atol=1e-12)

    def test_dm_matches_sv_noiseless(self):
        for circ in small_suite():
            p_sv = probabilities(circ)
            rho = density_matrix(circ, noise=None)
            p_dm = dm_diagonal(rho, circ.n_qubits)
            np.testing.assert_allclose(p_dm, p_sv, atol=1e-10)

    def test_readout_channel(self):
        p = np.array([1.0, 0.0])  # certain |0> on one qubit
        spec = NoiseSpec(qubits={0: QubitNoise(200, 100, readout_p01=0.1,
                                               readout_p10=0.3)})
        p2 = apply_readout(p, spec, {0: 0})
        np.testing.assert_allclose(p2, [0.9, 0.1], atol=1e-12)

    def test_relaxation_is_trace_preserving(self):
        spec = NoiseSpec(qubits={0: QubitNoise(200.0, 120.0)})
        ks = spec.relaxation_kraus(0, 500.0)
        total = sum(k.conj().T @ k for k in ks)
        np.testing.assert_allclose(total, np.eye(2), atol=1e-12)


class TestFrameTransform(unittest.TestCase):
    def test_operator_basis_reconstruction(self):
        """|a><b| = sum_s Tinv[s, 2a+b] |F_s><F_s|, exactly."""
        for a in range(2):
            for b in range(2):
                target = np.zeros((2, 2), dtype=complex)
                target[a, b] = 1.0
                rebuilt = sum(_TINV[s, 2 * a + b] * np.outer(_FRAME[s], _FRAME[s].conj())
                              for s in range(4))
                np.testing.assert_allclose(rebuilt, target, atol=1e-12)


class TestCandidates(unittest.TestCase):
    def test_validity_invariants(self):
        for circ in small_suite():
            cands = enumerate_candidates(circ, k_max=3)
            self.assertGreater(len(cands), 0)
            for cand in cands:
                validate_candidate(circ, cand)  # raises on violation
                self.assertEqual(cand.cut_type, "wire")

    def test_dedup(self):
        circ = ghz(5)
        cands = enumerate_candidates(circ, k_max=3)
        parts = [c.part_a for c in cands]
        self.assertEqual(len(parts), len(set(parts)))


class TestOracleExactness(unittest.TestCase):
    """Invariant 1: the exact path must be exact (fidelity 1 to 1e-10)."""

    def test_all_candidates_all_circuits(self):
        checked = 0
        for circ in small_suite():
            p_ideal = probabilities(circ)
            for cand in enumerate_candidates(circ, k_max=2):
                p_rec = reconstruct_noiseless(circ, cand)
                f = hellinger_fidelity(p_rec, p_ideal)
                self.assertLess(abs(f - 1.0), 1e-10,
                                f"oracle broken: F={f}, k={cand.k}")
                self.assertLess(total_variation(p_rec, p_ideal), 1e-9)
                checked += 1
        self.assertGreater(checked, 20)  # meaningful coverage, not vacuous pass


class TestNoisyLabeler(unittest.TestCase):
    def test_zero_noise_equivalence(self):
        """Invariant 2: density-picture path with no noise == oracle."""
        for circ in small_suite()[:4]:
            p_ideal = probabilities(circ)
            for cand in enumerate_candidates(circ, k_max=2)[:3]:
                p_noisy = reconstruct_noisy(circ, cand, noise=None)
                f = hellinger_fidelity(p_noisy, p_ideal)
                self.assertLess(abs(f - 1.0), 1e-9,
                                f"frame path broken: F={f}, k={cand.k}")

    def test_labels_informative(self):
        """Invariant 3: brisbane-class noise -> labels < 1 and varying."""
        snap = load_snapshot(FIXTURE)
        labels = []
        for circ in small_suite()[:5]:
            noise = build_noise_spec(snap, circ.n_qubits)
            for cand in enumerate_candidates(circ, k_max=2):
                rec = label_candidate(circ, cand, noise, check_oracle=False)
                labels.append(rec["pst_hellinger"])
                self.assertLess(rec["pst_hellinger"], 1.0)
                self.assertGreater(rec["pst_hellinger"], 0.3)
        self.assertGreater(len(labels), 10)
        self.assertGreater(float(np.std(labels)), 1e-4,
                           "labels are (near-)constant: no supervision signal")

    def test_noise_monotonicity(self):
        """Invariant 4: 10x worse T1/T2 must lower mean fidelity."""
        snap = load_snapshot(FIXTURE)
        circ = random_circuit(4, 3, 0.9, seed=11)[0]
        good = build_noise_spec(snap, 4)
        bad = build_noise_spec(snap, 4)
        for q in list(bad.qubits):
            qn = bad.qubits[q]
            bad.qubits[q] = QubitNoise(qn.t1_us / 10, qn.t2_us / 10,
                                       qn.readout_p01, qn.readout_p10)
        cands = enumerate_candidates(circ, k_max=2)[:4]
        f_good = np.mean([label_candidate(circ, c, good, check_oracle=False)
                          ["pst_hellinger"] for c in cands])
        f_bad = np.mean([label_candidate(circ, c, bad, check_oracle=False)
                         ["pst_hellinger"] for c in cands])
        self.assertLess(f_bad, f_good)


class TestGenerators(unittest.TestCase):
    def test_seed_reproducibility(self):
        for gen, args in [(random_circuit, (5, 3, 0.7)),
                          (twolocal_circuit, (5, 2)),
                          (qaoa_maxcut_circuit, (5, 1, 0.5))]:
            c1, m1 = gen(*args, 42)
            c2, m2 = gen(*args, 42)
            c3, _ = gen(*args, 43)
            self.assertEqual([(g.name, g.qubits, g.params) for g in c1.gates],
                             [(g.name, g.qubits, g.params) for g in c2.gates])
            self.assertEqual(m1, m2)
            self.assertNotEqual([(g.name, g.qubits, g.params) for g in c1.gates],
                                [(g.name, g.qubits, g.params) for g in c3.gates])


class TestCalibrationLoader(unittest.TestCase):
    def test_no_none_survives(self):
        snap = load_snapshot(FIXTURE)
        circ = twolocal_circuit(6, 2, 3)[0]
        feats = gate_node_features(circ, snap)
        self.assertEqual(len(feats), len(circ.gates))
        for row in feats:
            for key, val in row.items():
                self.assertIsNotNone(val, f"None in feature '{key}'")

    def test_both_aggregation_channels(self):
        snap = load_snapshot(FIXTURE)
        circ = Circuit(4).add("cx", (1, 3))
        row = gate_node_features(circ, snap)[0]
        self.assertLessEqual(row["t2_min"], row["t2_mean"])
        self.assertLessEqual(row["t1_min"], row["t1_mean"])
        self.assertEqual(row["gate_error"],
                         json.loads(FIXTURE.read_text())["default_error_2q"])

    def test_missing_key_rejected(self):
        import tempfile
        snap = json.loads(FIXTURE.read_text())
        snap["qubits"][2]["t2_us"] = None
        bad = Path(tempfile.gettempdir()) / "_quest_bad_snapshot.json"
        bad.write_text(json.dumps(snap))
        try:
            with self.assertRaises(ValueError):
                load_snapshot(bad)
        finally:
            bad.unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)
