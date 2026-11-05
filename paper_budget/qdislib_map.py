"""Map Qdislib find_cut results onto the thesis labeller (thesis environment).

Qdislib numbers gates from 1 in circuit order (observed in qdislib_smoke.py, 5 Nov 2026).
A gate cut is "NAME_k"; a wire cut is a pair (source, target) of gate ids: the wire of
the qubit the two gates share, cut right after the source gate. Each Qdislib cut is
rewritten in the form find_cuts returns (QPD gates and CutWire instructions), so the
same frozen mapper (analysis/findcuts_map.map_result) and labeller handle both baselines.
"""
import sys

from paper_budget.config import ROOT

sys.path.insert(0, str(ROOT / "stage2_predictor"))
sys.path.insert(0, str(ROOT / "stage2_predictor" / "analysis"))
import quest_phase2.data  # noqa: E402,F401  (puts the labeller package on sys.path)
from findcuts_map import map_result, to_qiskit  # noqa: E402


def gate_index(circ, gate_id):
    """'CX_11' -> 10, after checking that gate 10 of the circuit is a cx."""
    name, k = gate_id.rsplit("_", 1)
    i = int(k) - 1
    if not 0 <= i < len(circ.gates) or circ.gates[i].name != name.lower():
        raise ValueError(f"{gate_id} does not match gate {i} of the circuit")
    return i


def wire_cuts(circ, source, target):
    """('CX_8', 'CX_11') -> [(2, 7)]: qubit 2 cut right after gate 7."""
    i, j = gate_index(circ, source), gate_index(circ, target)
    shared = sorted(set(circ.gates[i].qubits) & set(circ.gates[j].qubits))
    if j <= i or not shared:
        raise ValueError(f"({source}, {target}) is not a wire between two gates")
    cuts = []
    for q in shared:
        nxt = next(k for k in range(i + 1, len(circ.gates)) if q in circ.gates[k].qubits)
        if nxt != j:
            raise ValueError(f"{target} is not the next gate on qubit {q} after {source}")
        cuts.append((q, i))
    return cuts


def normalise(result):
    """Any find_cut output -> (gate ids, wire pairs)."""
    if isinstance(result, dict):
        return list(result.get("gate", [])), [tuple(p) for p in result.get("wire", [])]
    gates = [x for x in result if isinstance(x, str)]
    wires = [tuple(x) for x in result if not isinstance(x, str)]
    return gates, wires


def cut_circuit(circ, result):
    """find_cut output -> Qiskit circuit with QPD gates and CutWire, as find_cuts returns."""
    from qiskit import QuantumCircuit
    from qiskit_addon_cutting.instructions import CutWire
    from qiskit_addon_cutting.qpd import TwoQubitQPDGate
    gate_ids, pairs = normalise(result)
    cut_gates = {gate_index(circ, g) for g in gate_ids}
    for i in cut_gates:
        if circ.gates[i].n_qubits != 2:
            raise ValueError(f"gate cut on a {circ.gates[i].n_qubits}-qubit gate ({i})")
    after = {}
    for s, t in pairs:
        for q, i in wire_cuts(circ, s, t):
            after.setdefault(i, set()).add(q)
    base = to_qiskit(circ)
    out = QuantumCircuit(circ.n_qubits)
    for i, inst in enumerate(base.data):
        qubits = [out.qubits[base.find_bit(b).index] for b in inst.qubits]
        op = TwoQubitQPDGate.from_instruction(inst.operation) if i in cut_gates else inst.operation
        out.append(op, qubits)
        for q in sorted(after.get(i, ())):
            out.append(CutWire(), [out.qubits[q]])
    return out


def map_qdislib(circ, result):
    """The same Mapping as for find_cuts: representable, reason, home_a, widths, ..."""
    return map_result(circ, cut_circuit(circ, result), {})


def pieces_after_gate_cuts(circ, gate_cut_indices):
    """Gate cuts only: logical qubits per piece (widest first) after removing the cut gates,
    and the cut gates whose two qubits stay in one piece (they separate nothing)."""
    parent = list(range(circ.n_qubits))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cut = set(gate_cut_indices)
    for i, g in enumerate(circ.gates):
        if g.n_qubits == 2 and i not in cut:
            parent[find(g.qubits[0])] = find(g.qubits[1])
    sizes = {}
    for q in range(circ.n_qubits):
        sizes[find(q)] = sizes.get(find(q), 0) + 1
    internal = sorted(i for i in cut if find(circ.gates[i].qubits[0]) == find(circ.gates[i].qubits[1]))
    return sorted(sizes.values(), reverse=True), internal
