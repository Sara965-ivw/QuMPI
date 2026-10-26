"""Fix B (23 Oct 2026): gate-cut reconstruction with any number of pieces, each simulated on its own."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "datagen"):
    sys.path.insert(0, str(p))


def _three_pieces():
    """9 qubits in three chains of 3, joined by one CX, one CZ and one RZZ (the cut gates)."""
    from quest_phase1.circuits import Circuit
    c = Circuit(9)
    for q in range(9):
        c.add("h", q); c.add("ry", q, 0.3 + 0.1 * q)
    for a, b in ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)):
        c.add("cx", (a, b))
    c.add("cx", (2, 3)); cut1 = len(c.gates) - 1
    c.add("cz", (5, 6)); cut2 = len(c.gates) - 1
    c.add("rzz", (8, 0), 0.7); cut3 = len(c.gates) - 1     # closes an odd cycle of pieces
    for q in range(9):
        c.add("rx", q, 0.2 * q)
    return c, [cut1, cut2, cut3]


def test_pieces_and_oracle_exact_with_three_pieces():
    from quest_phase1.labeling_v2 import make_device
    from quest_phase1.multipiece import gate_cut_pieces, reconstruct_multipiece
    from quest_phase1.observables import basket, expectations
    from quest_phase1.simulators import probabilities
    c, cuts = _three_pieces()
    assert gate_cut_pieces(c, cuts) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    rec = reconstruct_multipiece(make_device("fake_brisbane", seed=1234).ideal_copy(), c, cuts)
    p = probabilities(c)
    assert 0.5 * np.abs(rec["p"] - p).sum() < 1e-8
    spec = basket(c)
    assert np.max(np.abs(expectations(rec["p_quasi"], spec) - expectations(p, spec))) < 1e-8


def test_two_pieces_equal_the_two_side_labeller_with_noise():
    """With exactly two pieces the multipiece reconstruction must reproduce the existing
    two-side labeller (same templates, same noise) to numerical precision."""
    from quest_phase1.circuits import Circuit
    from quest_phase1.labeling_v2 import make_device
    from quest_phase1.mixed_cuts import MixedCut, plan_cut, reconstruct_plan
    from quest_phase1.multipiece import reconstruct_multipiece
    c = Circuit(6)
    for q in range(6):
        c.add("h", q)
    for a, b in ((0, 1), (1, 2), (3, 4), (4, 5)):
        c.add("cx", (a, b))
    c.add("cx", (2, 3)); c.add("rzz", (1, 4), 0.9)          # first qubit of each cut gate on side A
    for q in range(6):
        c.add("ry", q, 0.4)
    plan = plan_cut(c, MixedCut(frozenset({0, 1, 2}), (), 6))
    ctx = make_device("fake_brisbane", seed=1234)
    two = reconstruct_plan(ctx, c, plan)
    many = reconstruct_multipiece(ctx, c, plan.gate_cuts)
    assert np.max(np.abs(two["p_quasi"] - many["p_quasi"])) < 1e-10
    assert np.max(np.abs(two["p"] - many["p"])) < 1e-10


def test_fits_check_agrees_with_reconstruction():
    from quest_phase1.labeling_v2 import MappedWidthExceeded, make_device
    from quest_phase1.multipiece import multipiece_fits, reconstruct_multipiece
    c, cuts = _three_pieces()
    for ceiling, expect in ((3, False), (5, True)):    # a piece is 3 qubits; "M" slots add ancillas
        ctx = make_device("fake_brisbane", seed=1234)
        ctx.max_active_qubits = ceiling
        fits, det = multipiece_fits(ctx, c, cuts)
        assert fits == expect, (ceiling, det)
        try:
            reconstruct_multipiece(ctx.ideal_copy(), c, cuts)
            ok = True
        except MappedWidthExceeded:
            ok = False
        assert ok == fits, (ceiling, det)


def test_label_record():
    from quest_phase1.labeling_v2 import make_device
    from quest_phase1.multipiece import label_multipiece
    c, cuts = _three_pieces()
    rec = label_multipiece(make_device("fake_brisbane", seed=1234), c, cuts)
    assert rec["oracle_tvd"] < 1e-8 and 0.3 < rec["pst_hellinger"] < 1.0
    assert abs(rec["gamma2"] - 9 * 9 * (1 + 2 * abs(np.sin(0.7))) ** 2) < 1e-9
    assert rec["piece_widths"] == [3, 3, 3]


def test_inside_piece_cut_is_rejected():
    from quest_phase1.circuits import Circuit
    from quest_phase1.multipiece import reconstruct_multipiece
    from quest_phase1.labeling_v2 import make_device
    c = Circuit(3)
    c.add("cx", (0, 1)); c.add("cx", (1, 2)); c.add("cx", (0, 2))
    with pytest.raises(ValueError):
        reconstruct_multipiece(make_device("fake_brisbane", seed=1234), c, [1])
