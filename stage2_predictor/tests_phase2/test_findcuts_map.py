"""The find_cuts -> labeller mapping must describe exactly the cut find_cuts chose.
Checked with the labeller's noiseless oracle: an exact QPD reconstruction of the
mapped cut must reproduce the uncut distribution to machine precision."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "datagen", ROOT / "analysis"):
    sys.path.insert(0, str(p))
pytest.importorskip("qiskit_addon_cutting")


def _small_circuits():
    from quest_phase1 import generators as G
    return [G.twolocal_circuit(n_qubits=8, reps=2, seed=990901, entanglement="linear")[0],
            G.clustered_random_circuit(n_qubits=8, depth=3, n_blocks=2, p_bridge=0.6, seed=990902)[0]]


@pytest.mark.parametrize("k", [0, 1])
def test_mapped_cut_is_exact(k):
    import findcuts_map as F
    from quest_phase1.labeling_v2 import make_device
    from quest_phase1.mixed_cuts import plan_cut, reconstruct_plan
    from quest_phase1.simulators import probabilities
    circ = _small_circuits()[k]
    qc, meta = F.run_find_cuts(circ, qpu_width=5)
    m = F.map_result(circ, qc, meta)
    assert m.representable, m.reason
    plan = plan_cut(circ, F.as_mixed_cut(circ, m))
    assert sorted(plan.gate_cuts) == sorted(m.gate_cut_indices)
    assert abs(2 ** plan.log2_overhead - m.findcuts_gamma) < 1e-6 * m.findcuts_gamma
    rec = reconstruct_plan(make_device("fake_brisbane", seed=1234).ideal_copy(), circ, plan)
    assert 0.5 * np.abs(rec["p"] - probabilities(circ)).sum() < 1e-8


def test_separable_circuit_reports_no_cut():
    import findcuts_map as F
    from quest_phase1.circuits import Circuit
    c = Circuit(6)
    for a, b in ((0, 1), (1, 2), (3, 4), (4, 5)):
        c.add("h", a); c.add("cx", (a, b))
    qc, meta = F.run_find_cuts(c, qpu_width=3)
    m = F.map_result(c, qc, meta)
    assert m.no_cut_needed and not m.representable


def test_find_feasible_and_label_script_interface():
    """Decision A5 loop exists, returns a fitting cut, and the labelling script's imports resolve."""
    import importlib
    import findcuts_map as F
    from quest_phase1.labeling_v2 import make_device
    circ = _small_circuits()[0]
    res = F.find_feasible(make_device("fake_brisbane", seed=1234), circ, start=5)
    assert res["status"] == "ok" and res["width"] <= 5 and res["attempts"][-1]["fits"]
    for name in ("find_feasible", "matches_candidate", "routed_fits", "nocut_fits"):
        assert hasattr(F, name), name
    importlib.import_module("findcuts_label")


def test_routed_check_counts_measurement_templates():
    """Regression for the 23 Oct failure: a fragment that fits without a mid-circuit
    measurement but not with one (signed-measurement ancilla) must be rejected by
    routed_fits, exactly when the labeller itself rejects it."""
    import findcuts_map as F
    from quest_phase1.circuits import Circuit
    from quest_phase1.labeling_v2 import MappedWidthExceeded, make_device
    from quest_phase1.mixed_cuts import MixedCut, plan_cut, reconstruct_plan
    c = Circuit(8)
    for q in range(8):
        c.add("h", q)
    for q in range(7):
        c.add("cx", (q, q + 1))
    plan = plan_cut(c, MixedCut(frozenset(range(4)), (), 8))     # one gate cut on cx(3, 4)
    assert plan.g == 1 and plan.width_a == 4 and plan.width_b == 4
    for ceiling, expect_fit in ((4, False), (5, True)):
        ctx = make_device("fake_brisbane", seed=1234)
        ctx.max_active_qubits = ceiling
        fits, detail = F.routed_fits(ctx, c, plan)
        assert fits == expect_fit, (ceiling, detail)
        try:
            reconstruct_plan(ctx.ideal_copy(), c, plan)
            labeller_fits = True
        except MappedWidthExceeded:
            labeller_fits = False
        assert labeller_fits == fits, (ceiling, detail)


def test_fix_a_no_width_lowering_for_subcircuits_that_fit():
    """23 Oct failure: find_cuts returned three subcircuits that each fit, but two sides
    could not hold them, and the width was lowered until the search did not end.
    With fix A/B the result is accepted at the first width, as a multipiece cut."""
    import findcuts_map as F
    from quest_phase1.labeling_v2 import make_device
    from test_multipiece import _three_pieces
    c, _ = _three_pieces()
    ctx = make_device("fake_brisbane", seed=1234)
    ctx.max_active_qubits = 6
    res = F.find_feasible(ctx, c, start=3)
    assert res["status"] == "ok" and res["width"] == 3, res["attempts"]
    assert res["kind"] == "multipiece" and res["mapping"].n_fragments == 3
    assert len(res["attempts"]) == 1
