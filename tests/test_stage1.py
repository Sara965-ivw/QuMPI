"""Permanent regression tests for Stage 1.
These encode the hand-verified Phase 0 invariants; they must pass forever.
Run from repo root:  pytest tests/ -v
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "stage1_graph"))

import networkx as nx
from qiskit import QuantumCircuit
from dag_builder import to_dag
from dual_graph import build_dual_graphs


def ghz(n=4):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def weighted_circuit():
    """Interactions by construction: (0,1)x3, (1,2)x2, (2,3)x1."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1); qc.cx(1, 2); qc.cx(0, 1)
    qc.cx(2, 3); qc.cx(1, 2); qc.cx(0, 1)
    return qc


def test_ghz_topological_order_is_the_unique_chain():
    qc = ghz(4)
    ops = [(n.op.name, tuple(qc.find_bit(q).index for q in n.qargs))
           for n in to_dag(qc).topological_op_nodes()]
    assert ops == [("h", (0,)), ("cx", (0, 1)), ("cx", (1, 2)), ("cx", (2, 3))]


def test_gdg_is_always_acyclic():
    for qc in (ghz(4), ghz(8), weighted_circuit()):
        gdg, _ = build_dual_graphs(qc, to_dag(qc))
        assert nx.is_directed_acyclic_graph(gdg)


def test_qig_weights_match_hand_count():
    qc = weighted_circuit()
    _, qig = build_dual_graphs(qc, to_dag(qc))
    assert qig[0][1]["weight"] == 3
    assert qig[1][2]["weight"] == 2
    assert qig[2][3]["weight"] == 1
    assert qig.number_of_edges() == 3


def test_gdg_edge_count_matches_wire_arithmetic():
    """GHZ(4): per-wire consecutive-gate edges = 3+4+2+0... wait, per Phase 0
    hand count on weighted_circuit the total was 9."""
    qc = weighted_circuit()
    gdg, _ = build_dual_graphs(qc, to_dag(qc))
    assert gdg.number_of_nodes() == 7
    assert gdg.number_of_edges() == 9
