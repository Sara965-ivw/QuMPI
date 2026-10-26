"""Candidate cut-configuration enumeration (D2, wire cuts, dataset v1).

A candidate is an ancestor-closed gate bipartition (A, B): for each DAG node u,
A = ancestors(u) + {u}. Ancestor closure guarantees no DAG edge runs B -> A,
so the two-fragment quotient graph is acyclic *by construction* (the Roadmap
SS2.1-A causality requirement, satisfied trivially in the 2-fragment case).

Each qubit shared by A and B crosses the frontier exactly once (proof: if an
A-gate followed a B-gate on wire q, that B-gate would be an ancestor of an
A-gate and hence in A — contradiction). The number of wire cuts k therefore
equals the number of shared qubits.

Note recorded for the thesis (Ch. 5): a 'candidate cut location' in the sense
of a single DAG edge does not by itself define fragments; fragments come from
a gate partition, and the induced frontier may cross several wires. Candidates
here are therefore *cut configurations* with k >= 1, deduplicated, each tagged
with its k and per-wire cut positions.
"""
from __future__ import annotations

from dataclasses import dataclass

from .circuits import Circuit


@dataclass(frozen=True)
class CutCandidate:
    part_a: frozenset          # gate indices, ancestor-closed
    cut_qubits: tuple          # sorted physical qubits crossing the frontier
    cut_positions: tuple       # per cut qubit: (last A-gate idx, first B-gate idx)
    cut_type: str = "wire"     # dataset v1: wire cuts (D2); field kept for gate cuts

    @property
    def k(self) -> int:
        return len(self.cut_qubits)


def _ancestor_closures(circ: Circuit) -> list:
    preds = circ.predecessors()
    n = len(circ.gates)
    closures: list = [None] * n
    for i in range(n):  # topological: gate list order is temporal
        acc = set()
        for p in preds[i]:
            acc |= closures[p]
            acc.add(p)
        closures[i] = acc
    return closures


def enumerate_candidates(circ: Circuit, k_max: int = 2) -> list:
    """All distinct ancestor-closure bipartitions with 1 <= k <= k_max."""
    closures = _ancestor_closures(circ)
    n = len(circ.gates)
    seen: set = set()
    out: list = []
    for u in range(n):
        a = frozenset(closures[u] | {u})
        if len(a) == n or a in seen:
            continue
        seen.add(a)
        cand = _build_candidate(circ, a)
        if cand is not None and 1 <= cand.k <= k_max:
            out.append(cand)
    return out


def _build_candidate(circ: Circuit, part_a: frozenset):
    b = set(range(len(circ.gates))) - part_a
    qa, qb = circ.touched(part_a), circ.touched(b)
    shared = sorted(qa & qb)
    if not shared:
        return None  # disconnected split: no wire crosses, not a cut
    positions = []
    for q in shared:
        last_a = max(i for i in part_a if q in circ.gates[i].qubits)
        first_b = min(i for i in b if q in circ.gates[i].qubits)
        positions.append((last_a, first_b))
    return CutCandidate(part_a, tuple(shared), tuple(positions))


def validate_candidate(circ: Circuit, cand: CutCandidate) -> None:
    """Invariants: ancestor closure (no B->A edge); one crossing per shared qubit."""
    b = set(range(len(circ.gates))) - cand.part_a
    for u, v, _q in circ.dag_edges():
        if u in b and v in cand.part_a:
            raise AssertionError("B -> A edge: partition not ancestor-closed")
    for q, (last_a, first_b) in zip(cand.cut_qubits, cand.cut_positions):
        if not last_a < first_b:
            raise AssertionError(f"wire {q}: frontier positions out of order")
