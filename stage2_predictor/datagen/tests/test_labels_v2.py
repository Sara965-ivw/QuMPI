"""Regression tests for labelling scheme v2 (device-mapped fragments).

The informativeness test encodes the pilot finding of 10 Oct 2026: under v1
every label of a circuit equalled the noisy uncut fidelity to 1e-15. A v2
label set must vary across cut placements by far more than numerical error.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quest_phase1.candidates import enumerate_candidates
from quest_phase1.generators import random_circuit, twolocal_circuit
from quest_phase1.labeling_v2 import (label_candidate_v2, make_device,
                                      reconstruct_mapped, uncut_reference)
from quest_phase1.simulators import (hellinger_fidelity, probabilities,
                                     total_variation)

CTX = make_device("fake_brisbane", seed=1234)


class TestMappedOracle(unittest.TestCase):
    """Noise off: the mapped pipeline must reproduce the ideal distribution."""

    def test_oracle_all_candidates(self):
        ideal_ctx = CTX.ideal_copy()
        for circ in (random_circuit(5, 3, 0.8, seed=31)[0],
                     twolocal_circuit(5, 2, seed=32)[0]):
            p_ideal = probabilities(circ)
            for cand in enumerate_candidates(circ, k_max=2):
                p = reconstruct_mapped(ideal_ctx, circ, cand)["p"]
                self.assertLess(total_variation(p, p_ideal), 1e-8)

    def test_uncut_ideal_reference(self):
        circ = random_circuit(5, 3, 0.8, seed=33)[0]
        p = uncut_reference(CTX.ideal_copy(), circ)["p"]
        self.assertLess(total_variation(p, probabilities(circ)), 1e-8)


class TestInformativeness(unittest.TestCase):
    def test_labels_vary_across_cuts(self):
        circ = random_circuit(6, 4, 0.7, seed=34)[0]
        p_ideal = probabilities(circ)
        labels = [label_candidate_v2(CTX, circ, c, p_ideal, check_oracle=False)["pst_hellinger"]
                  for c in enumerate_candidates(circ, k_max=2)]
        self.assertGreater(max(labels) - min(labels), 1e-3,
                           "v2 labels do not depend on cut placement")
        self.assertTrue(all(0.0 < f < 1.0 for f in labels))


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_label(self):
        circ = twolocal_circuit(5, 2, seed=35)[0]
        cand = enumerate_candidates(circ, k_max=2)[0]
        a = label_candidate_v2(make_device(seed=7), circ, cand, check_oracle=False)
        b = label_candidate_v2(make_device(seed=7), circ, cand, check_oracle=False)
        self.assertEqual(a["pst_hellinger"], b["pst_hellinger"])
        self.assertEqual(a["phys_a"], b["phys_a"])


class TestReadoutConfusion(unittest.TestCase):
    def test_confusion_preserves_normalisation(self):
        from quest_phase1.labeling_v2 import _apply_confusion
        p = np.random.default_rng(0).random((2, 2, 2))
        p /= p.sum()
        q = _apply_confusion(p, {0: 0.03, 2: 0.1})
        self.assertAlmostEqual(q.sum(), 1.0, places=12)
        self.assertTrue((q >= 0).all())


if __name__ == "__main__":
    unittest.main()
