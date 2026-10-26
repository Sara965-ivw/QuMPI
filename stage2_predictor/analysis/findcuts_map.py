"""qiskit-addon-cutting find_cuts as an RQ1 baseline (Card A, decisions of 22 Oct 2026).

Proposal lines 309 and 569-570 name the overhead-minimising search of
qiskit-addon-cutting as an automated RQ1 baseline. find_cuts returns ONE cut
(not a ranking), so the comparison is on the cut each method selects: its
Hellinger fidelity under the same labeller (scheme v3.1, fake_brisbane,
transpiler seed 1234) and its top-1 regret against the oracle.

This module does the deterministic part:
  1. convert a QUEST circuit to a Qiskit QuantumCircuit;
  2. run find_cuts with the settings fixed in decision A3
     (qubits_per_subcircuit = 13, every other parameter at its default, fixed seed);
  3. map the returned cuts onto the labeller's representation (MixedCut): a
     bipartition into fragments A and B, gate cuts on every two-qubit gate that
     crosses the partition, and wire cuts that each move one qubit from A to B once.

A find_cuts result that does not fit that representation (more than two
fragments, a qubit cut more than once, wire cuts in both directions) is
reported with the reason. Decision A2 requires such cuts to be labelled too,
so the labeller is extended for them; the dry run measures how often this
happens before that extension is written.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

QPU_WIDTH = 13            # decision A3: same fragment-width limit as the candidate generator
FINDCUTS_SEED = 1234      # fixed seed for find_cuts' best-first search (reproducibility)


def to_qiskit(circ):
    """QUEST Circuit -> qiskit QuantumCircuit, gate for gate and in the same order."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(circ.n_qubits)
    for gt in circ.gates:
        name, qs, ps = gt.name, list(gt.qubits), list(gt.params)
        if name == "id":
            qc.id(qs[0])
        elif name in ("h", "x", "y", "z", "s", "t"):
            getattr(qc, name)(qs[0])
        elif name in ("rx", "ry", "rz"):
            getattr(qc, name)(ps[0], qs[0])
        elif name in ("cx", "cz"):
            getattr(qc, name)(qs[0], qs[1])
        elif name == "rzz":
            qc.rzz(ps[0], qs[0], qs[1])
        else:
            raise ValueError(f"gate {name} has no Qiskit mapping")
    return qc


def run_find_cuts(circ, qpu_width: int = QPU_WIDTH, seed: int = FINDCUTS_SEED):
    from qiskit_addon_cutting import DeviceConstraints, OptimizationParameters, find_cuts
    opt = OptimizationParameters(seed=seed)            # max_gamma, max_backjumps, gate_lo, wire_lo: defaults
    cons = DeviceConstraints(qubits_per_subcircuit=qpu_width)
    return find_cuts(to_qiskit(circ), opt, cons)


@dataclass
class Mapping:
    n_gate_cuts: int = 0
    n_wire_cuts: int = 0
    n_fragments: int = 0
    gate_cut_indices: list = field(default_factory=list)   # indices into circ.gates
    wire_cuts: list = field(default_factory=list)          # (qubit, last gate index before the cut)
    representable: bool = False
    reason: str = ""
    home_a: list = field(default_factory=list)
    width_a: int = 0
    width_b: int = 0
    findcuts_gamma: float = float("nan")                   # sampling overhead reported by find_cuts
    labeller_log2_overhead: float = float("nan")           # our overhead model for the same cut
    minimum_reached: Optional[bool] = None
    no_cut_needed: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def map_result(circ, cut_qc, meta: dict) -> Mapping:
    """Walk the cut circuit, rebuild fragments by union-find over qubit segments,
    and express the cut as a MixedCut when the labeller can represent it."""
    from qiskit_addon_cutting.instructions import CutWire
    from qiskit_addon_cutting.qpd import BaseQPDGate
    from quest_phase1.mixed_cuts import MixedCut, plan_cut

    m = Mapping(findcuts_gamma=float(meta.get("sampling_overhead", float("nan"))),
                minimum_reached=meta.get("minimum_reached"))
    seg = {q: 0 for q in range(circ.n_qubits)}          # current segment index per qubit
    last = {q: None for q in range(circ.n_qubits)}      # last original gate index on q
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    used = set()
    i = -1                                               # index into circ.gates
    for inst in cut_qc.data:
        op = inst.operation
        qs = [cut_qc.find_bit(b).index for b in inst.qubits]
        if isinstance(op, CutWire):
            q = qs[0]
            m.wire_cuts.append((q, last[q]))
            seg[q] += 1
            continue
        i += 1
        gt = circ.gates[i]
        if tuple(qs) != tuple(gt.qubits):
            raise AssertionError(f"gate order mismatch at {i}: {qs} vs {gt.qubits}")
        nodes = [(q, seg[q]) for q in qs]
        for nd in nodes:
            find(nd); used.add(nd)
        if isinstance(op, BaseQPDGate) or getattr(op, "name", "").startswith("qpd"):
            m.gate_cut_indices.append(i)
        elif len(nodes) == 2:
            union(nodes[0], nodes[1])
        for q in qs:
            last[q] = i
    if i != len(circ.gates) - 1:
        raise AssertionError(f"walked {i + 1} gates, circuit has {len(circ.gates)}")
    m.n_gate_cuts, m.n_wire_cuts = len(m.gate_cut_indices), len(m.wire_cuts)

    comps = {find(nd) for nd in used}
    m.n_fragments = len(comps)          # connected pieces after cutting (find_cuts' subcircuits)
    per_qubit = {q: [find((q, s)) for s in range(seg[q] + 1) if (q, s) in used] for q in range(circ.n_qubits)}
    if any(len(v) > 2 for v in per_qubit.values()):
        m.reason = "a qubit is wire-cut more than once"
        return m

    if m.n_gate_cuts + m.n_wire_cuts == 0:
        m.no_cut_needed = True
        m.reason = (f"no cut needed: the circuit already separates into {m.n_fragments} "
                    f"pieces of at most {QPU_WIDTH} qubits")
        return m

    # The labeller has two SIDES, A and B; a side may hold several disconnected pieces.
    # Constraints: each wire cut moves its qubit from a piece on A to a piece on B; each
    # gate cut joins pieces on different sides. Pieces linked by gate cuts form groups
    # whose internal colouring is fixed up to a flip (2-colouring); a group containing a
    # wire-cut piece has a fixed orientation. Free orientations are chosen to minimise
    # the wider side, so that a valid grouping within the 13-qubit limit is found if one exists.
    fixed: dict = {}
    for v in per_qubit.values():
        if len(v) == 2:
            for c, side in ((v[0], 0), (v[1], 1)):
                if fixed.setdefault(c, side) != side:
                    m.reason = "wire cuts require a piece on both sides"
                    return m
    adj: dict = {c: set() for c in comps}
    for gi in m.gate_cut_indices:
        q1, q2 = circ.gates[gi].qubits
        s1 = sum(1 for q, t in m.wire_cuts if q == q1 and gi > (-1 if t is None else t))
        s2 = sum(1 for q, t in m.wire_cuts if q == q2 and gi > (-1 if t is None else t))
        c1, c2 = find((q1, s1)), find((q2, s2))
        if c1 == c2:
            m.reason = "a gate cut joins a piece to itself"
            return m
        adj[c1].add(c2); adj[c2].add(c1)
    rel, groups = {}, []                     # rel[c] = colour relative to its group root
    for root in sorted(comps, key=str):
        if root in rel:
            continue
        rel[root], members, stack = 0, [root], [root]
        while stack:
            c = stack.pop()
            for d in adj[c]:
                if d not in rel:
                    rel[d] = 1 - rel[c]; members.append(d); stack.append(d)
                elif rel[d] == rel[c]:
                    m.reason = "gate cuts cannot be split into two sides (odd cycle)"
                    return m
        orient = {fixed[c] ^ rel[c] for c in members if c in fixed}
        if len(orient) > 1:
            m.reason = "wire-cut directions conflict with the gate cuts"
            return m
        groups.append((members, orient.pop() if orient else None))
    free = [k for k, (_, o) in enumerate(groups) if o is None]
    if len(free) > 16:
        m.reason = f"{len(free)} free pieces: grouping search too large"
        return m
    best = None
    for mask in range(2 ** len(free)):
        flips = {k: (mask >> j) & 1 for j, k in enumerate(free)}
        colour = {}
        for k, (members, o) in enumerate(groups):
            base = o if o is not None else flips[k]
            for c in members:
                colour[c] = base ^ rel[c]
        side_a = {q for q, v in per_qubit.items() if any(colour[c] == 0 for c in v)}
        side_b = {q for q, v in per_qubit.items() if any(colour[c] == 1 for c in v)}
        key = (max(len(side_a), len(side_b)), len(side_a))
        if best is None or key < best[0]:
            best = (key, colour)
    colour = {c: ("A" if x == 0 else "B") for c, x in best[1].items()}
    home_a = sorted(q for q, v in per_qubit.items() if v and colour[v[0]] == "A")
    cut = MixedCut(frozenset(home_a), tuple(sorted(m.wire_cuts)), circ.n_qubits)
    plan = plan_cut(circ, cut)
    if plan is None:
        m.reason = "plan_cut rejected the mapped cut"
        return m
    if sorted(plan.gate_cuts) != sorted(m.gate_cut_indices):
        m.reason = "gate cuts implied by the bipartition differ from find_cuts' gate cuts"
        return m
    m.home_a, m.width_a, m.width_b = home_a, plan.width_a, plan.width_b
    m.labeller_log2_overhead = plan.log2_overhead
    if max(plan.width_a, plan.width_b) > QPU_WIDTH:
        m.reason = f"fragment width {max(plan.width_a, plan.width_b)} > {QPU_WIDTH}"
        return m
    m.representable, m.reason = True, "ok"
    return m


def as_mixed_cut(circ, m: Mapping):
    from quest_phase1.mixed_cuts import MixedCut
    return MixedCut(frozenset(m.home_a), tuple(sorted(tuple(x) for x in m.wire_cuts)), circ.n_qubits)


# ---------------------------------------------------------------- feasibility (decision A5)
def routed_fits(ctx, circ, plan):
    """Map every fragment template exactly as reconstruct_plan does: the same order of
    QPD terms, the same zero-weight skips, and the layout of the first template reused
    for the later ones. A gate-cut term with a mid-circuit measurement adds one
    simulated qubit (the signed-measurement ancilla), so every template has to fit,
    not only the first. Returns (fits, per-side result: 'ok' or the active+measured count)."""
    import itertools
    import math as _m
    from qiskit.circuit import Parameter
    from quest_phase1.labeling_v2 import MappedWidthExceeded
    from quest_phase1.mixed_cuts import _RZZ_TERMS, _term_weights, build_fragments, map_fragment
    weights = {i: _term_weights(_m.pi / 2 if circ.gates[i].name in ("cx", "cz")
                                else float(circ.gates[i].params[0])) for i in plan.gate_cuts}
    params = {(i, sd): Parameter(f"g{i}{sd}") for i in plan.gate_cuts for sd in "AB"}
    done = {"A": set(), "B": set()}
    layout = {"A": None, "B": None}
    result = {"A": "ok", "B": "ok"}
    for tau in itertools.product(range(6), repeat=plan.g):
        w_tau = 1.0
        for i, t in zip(plan.gate_cuts, tau):
            w_tau *= weights[i][t]
        if w_tau == 0.0:
            continue
        ops = {i: (_RZZ_TERMS[t][1], _RZZ_TERMS[t][2]) for i, t in zip(plan.gate_cuts, tau)}
        keys = {"A": tuple(ops[i][0] == "M" for i in plan.gate_cuts),
                "B": tuple(ops[i][1] == "M" for i in plan.gate_cuts)}
        if all(keys[sd] in done[sd] for sd in "AB"):
            continue
        fa, fb = build_fragments(circ, plan, ops, params)
        for sd, n_loc, fr in (("A", plan.width_a, fa), ("B", plan.width_b, fb)):
            if keys[sd] in done[sd] or result[sd] != "ok":
                continue
            done[sd].add(keys[sd])
            try:
                mp = map_fragment(ctx, n_loc, fr, initial_layout=layout[sd])
                if layout[sd] is None:
                    layout[sd] = mp.initial
            except MappedWidthExceeded as e:
                result[sd] = str(e).split(" qubits")[0]
    return all(v == "ok" for v in result.values()), [result["A"], result["B"]]


def nocut_fits(ctx, circ):
    """Every piece of a separable circuit within the ceiling after routing?"""
    from quest_phase1.labeling_v2 import uncut_reference
    from quest_phase1.nocut import pieces, subcircuit
    act = [uncut_reference(ctx, subcircuit(circ, qs), simulate=False)["active_qubits"]
           for qs in pieces(circ)]
    return all(a <= ctx.max_active_qubits for a in act), act


def find_feasible(ctx, circ, start: int = QPU_WIDTH, seed: int = FINDCUTS_SEED):
    """Decisions A5 and A7 (fix A, 23 Oct 2026). Run find_cuts with
    qubits_per_subcircuit = start. Each subcircuit find_cuts returns is checked on
    its own, as it would be executed:
      * no cut needed      -> every independent piece (quest_phase1.nocut);
      * gate cuts only     -> every subcircuit (quest_phase1.multipiece);
      * with a wire cut    -> the two-side labeller, only if find_cuts returned exactly
                              two subcircuits (then the sides ARE the subcircuits).
    The width is lowered by one and find_cuts rerun from scratch ONLY when a single
    subcircuit is too wide after routing (decision A5). A result the labellers cannot
    represent stops the search and is reported."""
    from quest_phase1.mixed_cuts import plan_cut
    from quest_phase1.multipiece import multipiece_fits
    attempts = []
    for width in range(start, 1, -1):
        qc, md = run_find_cuts(circ, qpu_width=width, seed=seed)
        m = map_result(circ, qc, md)
        if m.no_cut_needed:
            kind = "nocut"
            ok, det = nocut_fits(ctx, circ)
        elif m.n_wire_cuts == 0:
            kind = "multipiece"
            ok, det = multipiece_fits(ctx, circ, m.gate_cut_indices)
        else:
            kind = "two_side"
            if not m.representable or m.n_fragments != 2:
                reason = m.reason if not m.representable else f"wire cut with {m.n_fragments} subcircuits"
                attempts.append({"width": width, "kind": kind, "status": "unrepresentable", "reason": reason})
                return {"status": "unrepresentable", "width": width, "kind": kind, "mapping": m,
                        "attempts": attempts}
            ok, det = routed_fits(ctx, circ, plan_cut(circ, as_mixed_cut(circ, m)))
        attempts.append({"width": width, "kind": kind, "fits": ok, "routed": det, "g": m.n_gate_cuts,
                         "w": m.n_wire_cuts, "subcircuits": m.n_fragments, "gamma2": m.findcuts_gamma})
        if ok:
            return {"status": "ok", "width": width, "kind": kind, "mapping": m, "attempts": attempts}
    return {"status": "infeasible", "width": None, "kind": None, "mapping": None, "attempts": attempts}


def matches_candidate(m: Mapping, rec: dict, n: int) -> bool:
    """Is the find_cuts cut one of the circuit's labelled candidates? Same cuts and the
    same bipartition (or its mirror image, for pure gate cuts)."""
    if m.no_cut_needed:
        return rec.get("cut_type") == "none"
    if sorted(rec["gate_cut_indices"]) != sorted(m.gate_cut_indices):
        return False
    if sorted(map(tuple, rec["wire_cuts"])) != sorted(map(tuple, m.wire_cuts)):
        return False
    a = set(rec["home_a"])
    return a == set(m.home_a) or (not m.wire_cuts and a == set(range(n)) - set(m.home_a))
