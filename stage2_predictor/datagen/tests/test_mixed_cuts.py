"""Regression tests for scheme v3: mixed wire + gate cuts, device-mapped.

Invariants: (1) the gate-cut decomposition is exact; (2) with noise disabled
the mapped reconstruction reproduces the ideal distribution for every cut
type; (3) every candidate strictly reduces the largest fragment width (the
corpus-probe failure of 10 Oct 2026); (4) noisy labels depend on the cut.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quest_phase1.generators import (clustered_random_circuit, qaoa_sbm_circuit,
                                     twolocal_circuit)
from quest_phase1.labeling_v2 import make_device
from quest_phase1.mixed_cuts import (_RZZ_TERMS, _term_weights, enumerate_mixed,
                                     gate_cut_gamma, label_circuit_v3, reconstruct_plan)
from quest_phase1.simulators import probabilities, total_variation

CTX = make_device("fake_brisbane", seed=1234)


class TestGateCutDecomposition(unittest.TestCase):
    def test_rzz_qpd_exact(self):
        P = [np.diag([1, 0]), np.diag([0, 1])]
        ops = {"I": np.eye(2), "Z": np.diag([1, -1]),
               "Sp": np.diag([np.exp(-1j * math.pi / 4), np.exp(1j * math.pi / 4)]),
               "Sm": np.diag([np.exp(1j * math.pi / 4), np.exp(-1j * math.pi / 4)])}

        def apply(op_a, op_b, rho):
            ka = [(1, ops[op_a])] if op_a != "M" else [((-1) ** m, P[m]) for m in (0, 1)]
            kb = [(1, ops[op_b])] if op_b != "M" else [((-1) ** m, P[m]) for m in (0, 1)]
            return sum(sa * sb * np.kron(a, b) @ rho @ np.kron(a, b).conj().T
                       for sa, a in ka for sb, b in kb)

        rng = np.random.default_rng(1)
        for theta in (0.3, math.pi / 2, 2.1):
            u = np.diag(np.exp(-1j * theta / 2 * np.array([1, -1, -1, 1])))
            v = rng.normal(size=4) + 1j * rng.normal(size=4)
            rho = np.outer(v, v.conj()) / np.vdot(v, v)
            rhs = sum(w * apply(a, b, rho)
                      for w, (_, a, b) in zip(_term_weights(theta), _RZZ_TERMS))
            np.testing.assert_allclose(u @ rho @ u.conj().T, rhs, atol=1e-12)
            self.assertAlmostEqual(sum(abs(w) for w in _term_weights(theta)),
                                   1 + 2 * abs(math.sin(theta)), places=12)
        self.assertAlmostEqual(gate_cut_gamma("cx", ()), 3.0)


class TestMappedOracle(unittest.TestCase):
    def test_all_cut_types_exact(self):
        ideal = CTX.ideal_copy()
        seen = set()
        for circ, _ in (clustered_random_circuit(6, 3, 2, 0.6, seed=41),
                        qaoa_sbm_circuit(6, 1, 2, 0.6, 2, seed=42),
                        twolocal_circuit(5, 2, seed=43)):
            p_ideal = probabilities(circ)
            per_type = {}
            for plan in enumerate_mixed(circ, g_max=2, w_max=1, max_width=5, n_random=10):
                if per_type.get(plan.cut_type, 0) >= 2:
                    continue
                per_type[plan.cut_type] = per_type.get(plan.cut_type, 0) + 1
                seen.add(plan.cut_type)
                p = reconstruct_plan(ideal, circ, plan)["p"]
                self.assertLess(total_variation(p, p_ideal), 1e-8, plan.cut_type)
        self.assertTrue({"gate", "mixed"} <= seen)


class TestFragmentsShrink(unittest.TestCase):
    def test_every_candidate_reduces_width(self):
        for n in (8, 14, 20):
            circ, _ = clustered_random_circuit(n, 4, 2, 0.4, seed=n)
            plans = enumerate_mixed(circ, g_max=2, w_max=1, max_width=13)
            self.assertTrue(plans, f"no candidates at n={n}")
            self.assertTrue(all(max(p.width_a, p.width_b) < n for p in plans))


class TestInformativeness(unittest.TestCase):
    def test_labels_vary_and_are_physical(self):
        circ, meta = clustered_random_circuit(6, 4, 2, 0.6, seed=44)
        out = label_circuit_v3(CTX, circ, meta, g_max=1, max_candidates=6,
                               oracle_fraction=0.2, sample_seed=1)
        f = [r["pst_hellinger"] for r in out["records"]]
        self.assertGreaterEqual(len(f), 3)
        self.assertTrue(all(0.0 < x < 1.0 for x in f))
        self.assertGreater(max(f) - min(f), 1e-3)


if __name__ == "__main__":
    unittest.main()
