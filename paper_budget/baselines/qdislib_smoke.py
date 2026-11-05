"""Smoke test of Qdislib find_cut on a circuit whose answer is known in advance.

Runs in the SEPARATE environment ~/venvs/qdislib-env (Qdislib 1.0.1 needs numpy<2.0).
Purpose: see Qdislib's output format before writing any converter.

Circuit: two 3-qubit clusters {0,1,2} and {3,4,5} joined by ONE gate, CX(2,3),
which is gate number 11 in Qdislib's 1-based numbering.
"""
import json

import Qdislib as qd
from qiskit import QuantumCircuit
from Qdislib.utils.backends import iter_gates


def test_circuit():
    qc = QuantumCircuit(6)
    for q in range(6):
        qc.h(q)                         # gates 1-6
    qc.cx(0, 1); qc.cx(1, 2)            # 7, 8
    qc.cx(3, 4); qc.cx(4, 5)            # 9, 10
    qc.cx(2, 3)                         # 11  <- the only bridge between the clusters
    qc.cx(0, 1); qc.cx(1, 2)            # 12, 13
    qc.cx(3, 4); qc.cx(4, 5)            # 14, 15
    qc.rz(0.3, 2); qc.rz(0.3, 3)        # 16, 17
    return qc


def main():
    qc = test_circuit()
    print("Qdislib", qd.__version__)
    print("\n--- gate numbering as Qdislib sees it ---")
    for i, (name, qubits) in enumerate(iter_gates(qc), start=1):
        print(f"{name}_{i}: qubits {qubits}")

    runs = {
        "gate_only": dict(max_qubits=3, wire_cut=False, gate_cut=True, seed=1234),
        "wire_only": dict(max_qubits=3, wire_cut=True, gate_cut=False, seed=1234),
        "default":   dict(max_qubits=3, seed=1234),
        "mixed":     dict(max_qubits=3, mixed=True, seed=1234),
        "top5":      dict(max_qubits=3, top_n=5, seed=1234),
        "sparsecut": dict(coupling_map=[(0, 1), (1, 2), (3, 4), (4, 5), (2, 5)], max_cuts=1),
    }
    out = {}
    for name, kw in runs.items():
        try:
            out[name] = qd.find_cut(qc, **kw)
        except Exception as exc:          # record the failure, keep testing the other modes
            out[name] = f"ERROR {type(exc).__name__}: {exc}"
        print(f"\n--- {name} ---\n{out[name]!r}")

    for name in ("gate_only", "wire_only"):
        if isinstance(out[name], list) and out[name]:
            try:
                print(f"\ncut_cost({name}) = {qd.cut_cost(out[name])!r}")
            except Exception as exc:
                print(f"\ncut_cost({name}) ERROR {type(exc).__name__}: {exc}")
    json.dumps(out, default=str)        # the converter will read results as JSON


if __name__ == "__main__":
    main()
