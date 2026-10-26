"""Regression tests for label scheme v3.1 (observable basket).

Invariants: (1) qubit/axis convention: X on qubit q flips exactly <Z_q>;
(2) GHZ expectations match theory; (3) expectations are linear, so they are
valid on quasi-distributions with negative entries; (4) with noise disabled the
QPD reconstruction reproduces every ideal expectation value for every cut type;
(5) noisy v3.1 records carry finite, physical observables; (6) the closed-form
finite-shot MSE agrees with Monte-Carlo sampling of the +-gamma QPD estimator.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quest_phase1.circuits import Circuit
from quest_phase1.generators import clustered_random_circuit, qaoa_sbm_circuit, twolocal_circuit
from quest_phase1.labeling_v2 import make_device
from quest_phase1.mixed_cuts import enumerate_mixed, label_circuit_v3, reconstruct_plan
from quest_phase1.observables import basket, expectations, mse
from quest_phase1.simulators import probabilities

CTX = make_device("fake_brisbane", seed=1234)


class TestConvention(unittest.TestCase):
    def test_x_flips_only_its_qubit(self):
        for q in range(4):
            c = Circuit(4).add("x", q)
            mu = expectations(probabilities(c), [(i,) for i in range(4)])
            expect = np.ones(4); expect[q] = -1
            np.testing.assert_allclose(mu, expect, atol=1e-12)

    def test_ghz(self):
        c = Circuit(4).add("h", 0)
        for i in range(3):
            c.add("cx", (i, i + 1))
        spec = basket(c)
        self.assertEqual(spec[:4], [(0,), (1,), (2,), (3,)])
        self.assertIn((0, 1), spec)
        self.assertEqual(spec[-1], (0, 1, 2, 3))
        mu = expectations(probabilities(c), spec)
        for S, m in zip(spec, mu):
            self.assertAlmostEqual(m, 0.0 if len(S) % 2 else 1.0, places=12)

    def test_linear_on_quasi_distributions(self):
        rng = np.random.default_rng(0)
        a, b = rng.random((2,) * 3), rng.random((2,) * 3)
        spec = [(0,), (1, 2), (0, 1, 2)]
        np.testing.assert_allclose(expectations(a - 0.3 * b, spec),
                                   expectations(a, spec) - 0.3 * expectations(b, spec),
                                   atol=1e-12)


class TestObservableOracle(unittest.TestCase):
    def test_ideal_reconstruction_preserves_expectations(self):
        ideal = CTX.ideal_copy()
        seen = set()
        for circ, _ in (clustered_random_circuit(6, 3, 2, 0.6, seed=41),
                        qaoa_sbm_circuit(6, 1, 2, 0.6, 2, seed=42),
                        twolocal_circuit(5, 2, seed=43)):
            spec = basket(circ)
            mu_ideal = expectations(probabilities(circ), spec)
            per_type = {}
            for plan in enumerate_mixed(circ, g_max=2, w_max=1, max_width=5, n_random=10):
                if per_type.get(plan.cut_type, 0) >= 2:
                    continue
                per_type[plan.cut_type] = per_type.get(plan.cut_type, 0) + 1
                seen.add(plan.cut_type)
                mu = expectations(reconstruct_plan(ideal, circ, plan)["p_quasi"], spec)
                np.testing.assert_allclose(mu, mu_ideal, atol=1e-8, err_msg=plan.cut_type)
        self.assertTrue({"gate", "mixed"} <= seen)


class TestNoisyRecords(unittest.TestCase):
    def test_v31_fields(self):
        circ, meta = clustered_random_circuit(6, 4, 2, 0.6, seed=44)
        out = label_circuit_v3(CTX, circ, meta, g_max=1, max_candidates=6,
                               oracle_fraction=0.2, sample_seed=1)
        k = len(out["obs_spec"])
        self.assertEqual(out["label_scheme"], "v3.1")
        self.assertEqual(len(out["obs_ideal"]), k)
        self.assertGreaterEqual(out["n_distinct_overhead"], 1)
        biases = []
        for r in out["records"]:
            mu = np.array(r["obs_mu"])
            self.assertEqual(len(mu), k)
            self.assertTrue(np.all(np.isfinite(mu)))
            self.assertTrue(np.all(np.abs(mu) <= 1.06))       # quasi-distribution norm tolerance
            self.assertTrue(np.all(r["gamma2"] - mu ** 2 > 0))  # per-shot variance positive
            biases.append(np.abs(mu - np.array(out["obs_ideal"])).max())
            if "oracle_obs_maxerr" in r:
                self.assertLess(r["oracle_obs_maxerr"], 1e-8)
        self.assertGreater(max(biases), 1e-4)                  # noise produces a real bias


class TestMSEFormula(unittest.TestCase):
    def test_closed_form_matches_sampling(self):
        rng = np.random.default_rng(7)
        gamma, mu, mu_ideal, N = 3.0, 0.62, 0.70, 400
        p_plus = (1 + mu / gamma) / 2                          # each sample is +-gamma
        est = gamma * (2 * rng.binomial(N, p_plus, size=200_000) - N) / N
        mc = np.mean((est - mu_ideal) ** 2)
        cf = float(mse([mu], [mu_ideal], gamma ** 2, N)[0])
        self.assertAlmostEqual(mc / cf, 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
