"""Converter tests on the smoke-test circuit; expected values are the Qdislib outputs
observed on 5 Nov 2026 (paper_budget/baselines/qdislib_smoke.py)."""
import pytest

pytest.importorskip("qiskit_addon_cutting")
from qiskit import QuantumCircuit  # noqa: E402

from paper_budget import qdislib_map as M  # noqa: E402
from quest_phase1.qiskit_bridge import from_qiskit  # noqa: E402

WIRE_ONLY_OBSERVED = [["CX_8", "CX_11"], ["CX_9", "CX_10"], ["CX_11", "CX_13"],
                      ["CX_11", "CX_14"], ["CX_12", "CX_13"]]


def smoke():
    qc = QuantumCircuit(6)
    for q in range(6):
        qc.h(q)
    qc.cx(0, 1); qc.cx(1, 2); qc.cx(3, 4); qc.cx(4, 5)
    qc.cx(2, 3)
    qc.cx(0, 1); qc.cx(1, 2); qc.cx(3, 4); qc.cx(4, 5)
    qc.rz(0.3, 2); qc.rz(0.3, 3)
    return from_qiskit(qc)


def test_gate_ids_are_one_based():
    c = smoke()
    assert M.gate_index(c, "CX_11") == 10
    assert c.gates[10].qubits == (2, 3)


def test_wrong_gate_name_is_rejected():
    with pytest.raises(ValueError):
        M.gate_index(smoke(), "CZ_11")


def test_wire_pair_maps_to_qubit_and_gate():
    assert M.wire_cuts(smoke(), "CX_8", "CX_11") == [(2, 7)]


def test_non_consecutive_pair_is_rejected():
    with pytest.raises(ValueError):          # on qubit 1, CX_12 lies between CX_8 and CX_13
        M.wire_cuts(smoke(), "CX_8", "CX_13")


def test_gate_cut_is_one_representable_gate_cut():
    m = M.map_qdislib(smoke(), ["CX_11"])
    assert m.gate_cut_indices == [10] and m.n_wire_cuts == 0
    assert m.n_fragments == 2 and m.representable


def test_observed_wire_only_cut_is_flagged_not_dropped():
    m = M.map_qdislib(smoke(), WIRE_ONLY_OBSERVED)
    assert m.n_wire_cuts == 5
    assert not m.representable and m.reason == "a qubit is wire-cut more than once"


def test_mixed_output_is_read():
    gates, wires = M.normalise({"gate": ["CX_10"], "wire": [["CX_9", "CX_11"]]})
    assert gates == ["CX_10"] and wires == [("CX_9", "CX_11")]


def test_pieces_after_the_bridge_cut():
    assert M.pieces_after_gate_cuts(smoke(), [10]) == ([3, 3], [])


def test_cut_inside_a_cluster_separates_nothing():
    assert M.pieces_after_gate_cuts(smoke(), [6]) == ([6], [6])


def test_internal_cut_is_found_next_to_a_separating_one():
    # CX_7 (gate 6) is internal: qubits 0 and 1 stay joined through CX_12 (gate 11)
    assert M.pieces_after_gate_cuts(smoke(), [10, 6]) == ([3, 3], [6])
