"""Mixed wire + gate cut candidates and their exact device-mapped labels (scheme v3).

Why this module exists (corpus probe, 10 Oct 2026)
---------------------------------------------------
The v1/v2 candidate generator produced only time-like wire cuts in which
fragment A is the past light cone of one gate. Every qubit touched by A has a
later gate, so k <= 2 wire cuts forces A onto at most two qubits and leaves
fragment B as wide as the whole circuit: no candidate ever reduced the
largest fragment width. At 14 or more qubits B exceeded the 13-qubit
simulation ceiling and no candidate was eligible.

Candidate definition (v3)
-------------------------
A candidate is a space-time bipartition of the circuit:
  * every qubit has a home side, A (set K) or B (set R);
  * up to `w_max` qubits C start on side A and move to side B after a chosen
    gate index t_q; each such move is a WIRE cut (Peng et al. 2020);
  * every two-qubit gate whose two qubits are on different sides at the time
    it acts is replaced by a GATE cut (Mitarai and Fujii 2021).
Pure wire, pure gate and mixed candidates all arise from this one definition
and are tagged cut_type in {"wire", "gate", "mixed"}. Both fragments are
strictly narrower than the circuit, by construction.

Gate-cut decomposition. CZ and CX are reduced to RZZ(pi/2) by local unitaries
that stay on their own side; RZZ(theta) is decomposed exactly as

  RZZ(.)RZZ^dag = c^2 [I,I] + s^2 [Z,Z]
                 + cs ([M,S] - [M,S^dag] + [S,M] - [S^dag,M]),

with c = cos(theta/2), s = sin(theta/2), S = RZ(pi/2) and M the signed
Z measurement rho -> P0 rho P0 - P1 rho P1 (verified numerically to 1e-16).
Sampling overhead gamma = 1 + 2|sin theta| (gamma = 3 for CZ/CX).

Signed measurements are simulated exactly by copying the measured physical
qubit onto a noiseless virtual ancilla (CNOT) and weighting the ancilla
outcome by (-1)^a; the measurement itself carries the same instrument noise
as a wire-cut measurement (relaxation for the readout duration plus an
assignment depolarizing channel).

Combination. For a joint gate-cut term tau with weight W_tau,
  p = sum_tau W_tau sum_{a,b} M_tau[zK,a,b] Wb_tau[a,b,zB],
where M_tau is the (signed) upstream density-matrix block on the wire-cut
qubits and Wb_tau the downstream response to frame inputs, exactly as in the
v1 wire combiner. Fragment runs are cached by their local operation tuple, so
a candidate with g gate cuts and w wire cuts costs 5^g upstream runs and
5^g * 4^w downstream runs.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .circuits import Circuit
from .combiner import _TINV
from .labeling_v2 import (DeviceContext, MappedWidthExceeded, _PREP_NATIVE,
                          _apply_confusion, _two_qubit_count)
from .observables import basket, expectations
from .simulators import hellinger_fidelity, probabilities, total_variation

LOG2_GAMMA2_WIRE = math.log2(16.0)
CUTTABLE = {"cx", "cz", "rzz"}

# RZZ cut terms: (weight function, op on side-A qubit, op on side-B qubit)
#   ops: "I", "Z", "Sp" (RZ(+pi/2)), "Sm" (RZ(-pi/2)), "M" (signed Z measurement)
_RZZ_TERMS = [("cc", "I", "I"), ("ss", "Z", "Z"),
              ("+cs", "M", "Sp"), ("-cs", "M", "Sm"),
              ("+cs", "Sp", "M"), ("-cs", "Sm", "M")]


def _term_weights(theta: float) -> list:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    table = {"cc": c * c, "ss": s * s, "+cs": c * s, "-cs": -c * s}
    return [table[w] for w, _, _ in _RZZ_TERMS]


def gate_cut_gamma(gate_name: str, params: tuple) -> float:
    """Optimal single-gate-cut overhead for RZZ(theta): 1 + 2|sin theta| (3 for CX/CZ)."""
    theta = math.pi / 2 if gate_name in ("cx", "cz") else float(params[0])
    return 1.0 + 2.0 * abs(math.sin(theta))


# ------------------------------------------------------------------ candidates
@dataclass(frozen=True)
class MixedCut:
    home_a: frozenset            # qubits whose home side is A (K plus wire-cut qubits)
    wire_cuts: tuple             # ((qubit, last gate index on side A), ...) sorted
    n_qubits: int

    @property
    def wire_qubits(self) -> tuple:
        return tuple(q for q, _ in self.wire_cuts)

    def side(self, q: int, gate_index: int) -> str:
        for wq, t in self.wire_cuts:
            if wq == q:
                return "A" if gate_index <= t else "B"
        return "A" if q in self.home_a else "B"


@dataclass
class CutPlan:
    cut: MixedCut
    k_qubits: list               # A-only outputs
    c_qubits: list               # wire-cut qubits (A then B)
    r_qubits: list               # B-only outputs
    gate_cuts: list              # gate indices replaced by gate cuts
    log2_overhead: float
    cut_type: str

    @property
    def g(self) -> int:
        return len(self.gate_cuts)

    @property
    def w(self) -> int:
        return len(self.c_qubits)

    @property
    def width_a(self) -> int:
        return len(self.k_qubits) + len(self.c_qubits)

    @property
    def width_b(self) -> int:
        return len(self.c_qubits) + len(self.r_qubits)


def plan_cut(circ: Circuit, cut: MixedCut) -> Optional[CutPlan]:
    """Validate a candidate and derive its fragment structure; None if invalid."""
    wq = set(cut.wire_qubits)
    gate_cuts, log2 = [], LOG2_GAMMA2_WIRE * len(wq)
    used_a, used_b = set(), set()
    for i, gt in enumerate(circ.gates):
        sides = [cut.side(q, i) for q in gt.qubits]
        for q, sd in zip(gt.qubits, sides):
            (used_a if sd == "A" else used_b).add(q)
        if len(set(sides)) == 2:
            if gt.name not in CUTTABLE:
                return None
            gate_cuts.append(i)
            log2 += 2 * math.log2(gate_cut_gamma(gt.name, gt.params))
    # a wire cut must actually carry state from A to B
    for q, t in cut.wire_cuts:
        on_q = [i for i, gt in enumerate(circ.gates) if q in gt.qubits]
        if not on_q or t < on_q[0] or t >= on_q[-1]:
            return None
    k = sorted(q for q in range(cut.n_qubits) if q not in wq and q in cut.home_a)
    r = sorted(q for q in range(cut.n_qubits) if q not in wq and q not in cut.home_a)
    c = sorted(wq)
    if not (k or c) or not (r or c):
        return None
    if not gate_cuts and not wq:
        return None   # disconnected: not a cut
    cut_type = "mixed" if (gate_cuts and wq) else ("gate" if gate_cuts else "wire")
    return CutPlan(cut, k, c, r, gate_cuts, log2, cut_type)


def enumerate_mixed(circ: Circuit, g_max: int = 3, w_max: int = 1,
                    max_width: int = 13, n_random: int = 40, seed: int = 0) -> list:
    """Candidate space-time bipartitions with g <= g_max gate cuts, w <= w_max
    wire cuts and both fragments <= max_width and narrower than the circuit.

    Sources (all deduplicated): Kernighan-Lin bisections from several random
    starts (structurally good cuts), random balanced and unbalanced partitions
    (diversity, so the label set contains poor cuts too), and for each
    partition the single wire cut on a boundary qubit that most reduces the
    number of gate cuts (mixed and pure wire candidates)."""
    import networkx as nx
    rng = np.random.default_rng(seed)
    n = circ.n_qubits
    qig = nx.Graph()
    qig.add_nodes_from(range(n))
    for gt in circ.gates:
        if gt.n_qubits == 2:
            a, b = gt.qubits
            w = qig[a][b]["weight"] + 1 if qig.has_edge(a, b) else 1
            qig.add_edge(a, b, weight=w)

    parts = set()
    for s in range(12):
        a, _ = nx.algorithms.community.kernighan_lin_bisection(qig, weight="weight", seed=int(s))
        parts.add(frozenset(a))
    for _ in range(n_random):
        size = int(rng.integers(max(1, n - max_width), min(max_width, n - 1) + 1))
        parts.add(frozenset(rng.choice(n, size=size, replace=False).tolist()))
    # contiguous blocks (natural for linear / clustered layouts)
    for size in range(max(1, n - max_width), min(max_width, n - 1) + 1):
        parts.add(frozenset(range(size)))

    out, seen = [], set()

    def consider(cut):
        key = (cut.home_a if 0 in cut.home_a else frozenset(range(n)) - cut.home_a,
               cut.wire_cuts)
        if key in seen:
            return
        seen.add(key)
        plan = plan_cut(circ, cut)
        if plan is None or plan.g > g_max or plan.w > w_max:
            return
        if max(plan.width_a, plan.width_b) > max_width or max(plan.width_a, plan.width_b) >= n:
            return
        out.append(plan)

    for home_a in parts:
        if not home_a or len(home_a) == n:
            continue
        base = MixedCut(frozenset(home_a), (), n)
        consider(base)
        if w_max >= 1:
            # best single wire cut: move one boundary qubit from A to B mid-circuit
            for q in sorted(home_a):
                on_q = [i for i, gt in enumerate(circ.gates) if q in gt.qubits]
                for t in on_q[:-1]:
                    consider(MixedCut(frozenset(home_a), ((q, t),), n))
    return out


# ------------------------------------------------------------ fragment builder
def _local_rzz_frame(gt) -> tuple:
    """(theta, pre ops, post ops) reducing a cuttable gate to RZZ(theta).
    ops are lists of (qubit, name, param)."""
    if gt.name == "rzz":
        return float(gt.params[0]), [], []
    if gt.name == "cz":   # CZ ~ (Rz(-pi/2) x Rz(-pi/2)) RZZ(pi/2)
        a, b = gt.qubits
        return math.pi / 2, [], [(a, "rz", -math.pi / 2), (b, "rz", -math.pi / 2)]
    if gt.name == "cx":   # CX = (I x H) CZ (I x H)
        c, t = gt.qubits
        return (math.pi / 2, [(t, "h", None)],
                [(c, "rz", -math.pi / 2), (t, "rz", -math.pi / 2), (t, "h", None)])
    raise ValueError(gt.name)


_SLOT_ANGLE = {"I": 0.0, "Z": math.pi, "Sp": math.pi / 2, "Sm": -math.pi / 2}


def build_fragments(circ: Circuit, plan: CutPlan, ops: dict, params: Optional[dict] = None) -> tuple:
    """Operation lists for fragments A and B given a choice of local op per
    gate cut. ops: {gate_index: (opA, opB)}. Returns (frag_a, frag_b) with
    entries (name, local qubits tuple, params) where name may be 'mz'."""
    cut = plan.cut
    map_a = {q: i for i, q in enumerate(plan.k_qubits + plan.c_qubits)}
    map_b = {q: i for i, q in enumerate(plan.c_qubits + plan.r_qubits)}
    fa, fb = [], []

    def emit(side, q, name, param):
        (fa if side == "A" else fb).append(
            (name, ((map_a if side == "A" else map_b)[q],), () if param is None else (param,)))

    for i, gt in enumerate(circ.gates):
        sides = [cut.side(q, i) for q in gt.qubits]
        if i not in ops:
            side = sides[0]
            m = map_a if side == "A" else map_b
            (fa if side == "A" else fb).append((gt.name, tuple(m[q] for q in gt.qubits), gt.params))
            continue
        theta, pre, post = _local_rzz_frame(gt)
        side_of = dict(zip(gt.qubits, sides))
        for q, name, p in pre:
            emit(side_of[q], q, name, p)
        qa = next(q for q in gt.qubits if side_of[q] == "A")
        qb = next(q for q in gt.qubits if side_of[q] == "B")
        for q, op, side_key in ((qa, ops[i][0], "A"), (qb, ops[i][1], "B")):
            sd = side_of[q]
            if op == "M":
                emit(sd, q, "mz", None)
            elif params is not None:
                # one RZ slot per cut and side: I, Z, S, S^dag differ only in angle
                # (RZ(pi) = -iZ: a global phase), so one transpiled template serves all four
                emit(sd, q, "rz", params[(i, side_key)])
            elif op != "I":
                emit(sd, q, "rz", _SLOT_ANGLE[op])
        for q, name, p in post:
            emit(side_of[q], q, name, p)
    return fa, fb


# ------------------------------------------------------------- mapped execution
def _qc_from_ops(n_local: int, frag: list):
    from qiskit import ClassicalRegister, QuantumCircuit
    n_meas = sum(1 for name, _, _ in frag if name == "mz")
    qc = QuantumCircuit(n_local)
    if n_meas:
        qc.add_register(ClassicalRegister(n_meas, "m"))
    j = 0
    for name, qs, params in frag:
        if name == "mz":
            qc.barrier()
            qc.measure(qs[0], j)
            qc.barrier()
            j += 1
        else:
            getattr(qc, name)(*params, *qs)
    return qc


def _instrument(ctx: DeviceContext, p: int):
    from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error
    err = depolarizing_error(min(1.0, 2.0 * ctx.readout_eps[p]), 1)
    t1, t2 = ctx.t1[p], ctx.t2[p]
    if ctx.meas_duration[p] > 0 and math.isfinite(t1):
        err = thermal_relaxation_error(t1, min(t2, 2 * t1), ctx.meas_duration[p]).compose(err)
    return err


@dataclass
class Mapped:
    tqc: object
    initial: list
    final: list
    n_meas: int
    n_2q_logical: int
    n_2q_mapped: int
    depth: int


def map_fragment(ctx: DeviceContext, n_local: int, frag: list,
                 initial_layout: Optional[list] = None) -> Mapped:
    from qiskit import transpile
    qc = _qc_from_ops(n_local, frag)
    tqc = transpile(qc, backend=ctx.backend, initial_layout=initial_layout,
                    optimization_level=ctx.opt_level, seed_transpiler=ctx.seed)
    initial = list(tqc.layout.initial_index_layout(filter_ancillas=True))[:n_local]
    final = list(tqc.layout.final_index_layout(filter_ancillas=True))[:n_local]
    n_meas = sum(1 for inst in tqc.data if inst.operation.name == "measure")
    active = {tqc.find_bit(q).index for inst in tqc.data
              if inst.operation.name not in ("barrier", "delay") for q in inst.qubits}
    if len(active) + n_meas > ctx.max_active_qubits:
        raise MappedWidthExceeded(f"{len(active)}+{n_meas} qubits (ceiling {ctx.max_active_qubits})")
    return Mapped(tqc, initial, final, n_meas, _two_qubit_count(qc), _two_qubit_count(tqc),
                  tqc.depth())


def simulate_mapped(ctx: DeviceContext, m: Mapped, n_local: int,
                    prep_phys: Optional[dict] = None, instrument_local: tuple = ()) -> np.ndarray:
    """Signed reduced density matrix over the local qubits (numpy convention).
    Mid-circuit measurements become instrument noise + CNOT onto a noiseless
    virtual ancilla; the ancilla outcome a contributes the sign (-1)^a."""
    from qiskit import QuantumCircuit
    nq = m.tqc.num_qubits
    run = QuantumCircuit(nq + m.n_meas)
    if prep_phys:
        for pq, s in prep_phys.items():
            for name, theta in _PREP_NATIVE[s]:
                run.rz(theta, pq) if name == "rz" else getattr(run, name)(pq)
    j = 0
    for inst in m.tqc.data:
        qs = [m.tqc.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "measure":
            if ctx.noisy:
                run.append(_instrument(ctx, qs[0]).to_instruction(), [qs[0]])
            run.cx(qs[0], nq + j)
            j += 1
        elif inst.operation.name == "barrier":
            continue
        else:
            run.append(inst.operation, qs)
    if ctx.noisy:
        for i in instrument_local:
            run.append(_instrument(ctx, m.final[i]).to_instruction(), [m.final[i]])
    keep = [m.final[i] for i in range(n_local)] + [nq + a for a in range(m.n_meas)]
    run.save_density_matrix(qubits=keep)
    res = ctx.simulator().run(run).result()
    tot = n_local + m.n_meas
    dm = np.asarray(res.data()["density_matrix"]).reshape((2,) * (2 * tot))
    perm = list(reversed(range(tot))) + [tot + a for a in reversed(range(tot))]
    dm = np.transpose(dm, perm)       # axes: local..., anc..., local..., anc...
    if m.n_meas:
        d = 2 ** n_local
        dm = dm.reshape(d, 2 ** m.n_meas, d, 2 ** m.n_meas)
        signs = np.array([(-1) ** bin(a).count("1") for a in range(2 ** m.n_meas)])
        dm = np.einsum("iaja,a->ij", dm, signs).reshape((2,) * (2 * n_local))
    return dm


def _diag(rho, n):
    return np.real(np.einsum(rho.reshape(2 ** n, 2 ** n), [0, 0], [0])).reshape((2,) * n)


def reconstruct_plan(ctx: DeviceContext, circ: Circuit, plan: CutPlan) -> dict:
    nk, nc, nr = len(plan.k_qubits), len(plan.c_qubits), len(plan.r_qubits)
    na, nb = nk + nc, nc + nr
    weights = {i: _term_weights(math.pi / 2 if circ.gates[i].name in ("cx", "cz")
                                else float(circ.gates[i].params[0]))
               for i in plan.gate_cuts}
    from qiskit.circuit import Parameter
    params = {(i, sd): Parameter(f"g{i}{sd}") for i in plan.gate_cuts for sd in "AB"}
    cache_a, cache_b, tmpl_a, tmpl_b = {}, {}, {}, {}
    layout_a = layout_b = None
    p = np.zeros((2 ** nk, 2 ** nb))
    stats = {"n_2q_logical": 0, "n_2q_mapped": 0, "depth_a": 0, "depth_b": 0, "transpiles": 0}

    def bound(m, side, key):
        values = {params[(i, side)]: _SLOT_ANGLE[op]
                  for i, op in zip(plan.gate_cuts, key) if op != "M"}
        live = {k: v for k, v in values.items() if k in m.tqc.parameters}
        tqc = m.tqc.assign_parameters(live) if live else m.tqc
        return Mapped(tqc, m.initial, m.final, m.n_meas, m.n_2q_logical, m.n_2q_mapped, m.depth)

    for tau in itertools.product(range(6), repeat=plan.g):
        w_tau = 1.0
        for i, t in zip(plan.gate_cuts, tau):
            w_tau *= weights[i][t]
        if w_tau == 0.0:
            continue
        ops = {i: (_RZZ_TERMS[t][1], _RZZ_TERMS[t][2]) for i, t in zip(plan.gate_cuts, tau)}
        key_a = tuple(ops[i][0] for i in plan.gate_cuts)
        key_b = tuple(ops[i][1] for i in plan.gate_cuts)
        struct_a = tuple(o == "M" for o in key_a)
        struct_b = tuple(o == "M" for o in key_b)
        if struct_a not in tmpl_a or struct_b not in tmpl_b:
            fa, fb = build_fragments(circ, plan, ops, params)
        if struct_a not in tmpl_a:
            tmpl_a[struct_a] = map_fragment(ctx, na, fa, initial_layout=layout_a)
            stats["transpiles"] += 1
            if layout_a is None:
                ma0 = tmpl_a[struct_a]
                layout_a = ma0.initial
                stats.update(n_2q_logical=ma0.n_2q_logical, depth_a=ma0.depth,
                             n_2q_mapped=ma0.n_2q_mapped)
        if struct_b not in tmpl_b:
            tmpl_b[struct_b] = map_fragment(ctx, nb, fb, initial_layout=layout_b)
            stats["transpiles"] += 1
            if layout_b is None:
                mb0 = tmpl_b[struct_b]
                layout_b = mb0.initial
                stats["n_2q_logical"] += mb0.n_2q_logical
                stats["n_2q_mapped"] += mb0.n_2q_mapped
                stats["depth_b"] = mb0.depth
        if key_a not in cache_a:
            ma = bound(tmpl_a[struct_a], "A", key_a)
            rho = simulate_mapped(ctx, ma, na, instrument_local=tuple(range(nk, na)))
            mm = np.einsum("iaib->iab", rho.reshape(2 ** nk, 2 ** nc, 2 ** nk, 2 ** nc))
            cache_a[key_a] = (mm, ma.final)
        if key_b not in cache_b:
            mb = bound(tmpl_b[struct_b], "B", key_b)
            q_frames = np.empty((4,) * nc + (2 ** nb,))
            for s_flat in range(4 ** nc):
                s = [(s_flat // 4 ** (nc - 1 - j)) % 4 for j in range(nc)]
                prep = {mb.initial[j]: s[j] for j in range(nc) if s[j] != 0}
                q_frames[tuple(s)] = _diag(simulate_mapped(ctx, mb, nb, prep_phys=prep),
                                           nb).reshape(-1)
            wv = q_frames.astype(complex)
            for j in range(nc):
                wv = np.moveaxis(np.tensordot(_TINV.T, wv, axes=(1, j)), 0, j)
            wv = wv.reshape((2, 2) * nc + (2 ** nb,))
            wv = np.transpose(wv, [2 * j for j in range(nc)] + [2 * j + 1 for j in range(nc)]
                              + [2 * nc]).reshape(2 ** nc, 2 ** nc, 2 ** nb)
            cache_b[key_b] = (wv, mb.final)
        mm, fin_a = cache_a[key_a]
        wv, fin_b = cache_b[key_b]
        term = np.einsum("iab,abz->iz", mm, wv)
        if np.max(np.abs(term.imag)) > 1e-8:
            raise AssertionError("imaginary residue in mixed combiner")
        p += w_tau * term.real
    # Exactness: with ideal operations the QPD reconstruction is a normalised
    # distribution (checked tightly by the noiseless oracle). With noisy
    # operations the gate-cut terms are not exactly trace-balanced (e.g. the
    # +S and -S branches transpile to slightly different native sequences), so
    # the reconstruction is a quasi-distribution with small norm error and
    # negative mass, exactly as on hardware. Both are recorded, then the
    # standard projection (clip negatives, renormalise) gives the label.
    total = float(p.sum())
    neg_mass = float(-p[p < 0].sum())
    tol = 0.05 if ctx.noisy else 1e-6
    if abs(total - 1.0) > tol:
        raise AssertionError(f"mixed combiner norm residue: {total}")
    # reorder axes (K, C, R) -> logical qubit order
    order = plan.k_qubits + plan.c_qubits + plan.r_qubits
    phys = {q: fin_a[i] for i, q in enumerate(plan.k_qubits)}
    phys.update({q: fin_b[i] for i, q in enumerate(plan.c_qubits + plan.r_qubits)})

    def _to_logical(arr):
        arr = np.transpose(arr.reshape((2,) * (nk + nb)), np.argsort(order))
        if ctx.noisy:   # readout confusion is linear, so it commutes with the QPD sum
            arr = _apply_confusion(arr, {q: ctx.readout_eps[pq] for q, pq in phys.items()})
        return arr

    # v3.1: the UNCLIPPED quasi-distribution is what the finite-shot QPD
    # estimator converges to; expectation values are computed from it.
    p_quasi = _to_logical(p.copy())
    p = np.clip(p, 0.0, None)
    p_full = _to_logical(p / p.sum())
    return {"p": p_full, "p_quasi": p_quasi, "phys_a": [int(x) for x in layout_a], "phys_b": [int(x) for x in layout_b],
            "runs_a": len(cache_a), "runs_b": len(cache_b) * 4 ** nc,
            "norm_residue": total - 1.0, "negative_mass": neg_mass, **stats}


# --------------------------------------------------------------------- labels
def label_plan(ctx: DeviceContext, circ: Circuit, plan: CutPlan, p_ideal=None,
               f_uncut: Optional[float] = None, check_oracle: bool = True,
               obs_spec: Optional[list] = None, obs_ideal=None) -> dict:
    if p_ideal is None:
        p_ideal = probabilities(circ)
    t0 = time.perf_counter()
    out = {"cut_type": plan.cut_type, "g": plan.g, "w": plan.w, "k": plan.g + plan.w,
           "home_a": sorted(plan.cut.home_a), "wire_cuts": [list(x) for x in plan.cut.wire_cuts],
           "gate_cut_indices": plan.gate_cuts, "width_a": plan.width_a, "width_b": plan.width_b,
           "log2_sampling_overhead": plan.log2_overhead, "label_scheme": "v3.1"}
    if obs_spec is None:
        obs_spec = basket(circ)
    if obs_ideal is None:
        obs_ideal = expectations(p_ideal, obs_spec)
    if check_oracle:
        ideal = reconstruct_plan(ctx.ideal_copy(), circ, plan)
        tvd = total_variation(ideal["p"], p_ideal)
        if tvd > 1e-8:
            raise AssertionError(f"mixed-cut oracle violated: TVD={tvd:.2e}")
        dmu = float(np.max(np.abs(expectations(ideal["p_quasi"], obs_spec) - obs_ideal)))
        if dmu > 1e-8:
            raise AssertionError(f"observable oracle violated: max |dmu|={dmu:.2e}")
        out["oracle_tvd"] = float(tvd)
        out["oracle_obs_maxerr"] = dmu
    rec = reconstruct_plan(ctx, circ, plan)
    out["obs_mu"] = [float(x) for x in expectations(rec["p_quasi"], obs_spec)]
    out["gamma2"] = float(2.0 ** plan.log2_overhead)
    out["pst_hellinger"] = float(hellinger_fidelity(rec["p"], p_ideal))
    out["tv_distance"] = float(total_variation(rec["p"], p_ideal))
    if f_uncut is not None:
        out["f_uncut"] = float(f_uncut)
        out["delta_vs_uncut"] = out["pst_hellinger"] - float(f_uncut)
    for key in ("phys_a", "phys_b", "runs_a", "runs_b", "n_2q_logical", "n_2q_mapped",
                "depth_a", "depth_b", "norm_residue", "negative_mass"):
        out[key] = rec[key]
    out["routing_overhead_2q"] = rec["n_2q_mapped"] - rec["n_2q_logical"]
    out["label_wall_s"] = time.perf_counter() - t0
    out.update(device=ctx.name, calibration_date=ctx.calibration_date,
               transpiler_seed=ctx.seed, opt_level=ctx.opt_level)
    return out


def label_circuit_v3(ctx: DeviceContext, circ: Circuit, meta: dict, g_max: int = 2,
                     w_max: int = 1, max_width: int = 13, max_candidates: Optional[int] = 30,
                     oracle_fraction: float = 0.1, sample_seed: int = 0,
                     uncut_max_qubits: int = 13) -> dict:
    """Label one circuit under scheme v3. Sampling is stratified by cut type:
    every pure wire and pure gate candidate is kept (they are rare), the rest
    of the budget is filled with a seeded sample of mixed candidates. Every
    exclusion is counted in the output for the datasheet."""
    from .labeling_v2 import device_node_features, uncut_reference
    rng = np.random.default_rng(sample_seed)
    p_ideal = probabilities(circ)
    ref = uncut_reference(ctx, circ, simulate=circ.n_qubits <= uncut_max_qubits)
    f_uncut = hellinger_fidelity(ref["p"], p_ideal) if ref["p"] is not None else None
    obs_spec = basket(circ)
    obs_ideal = expectations(p_ideal, obs_spec)
    obs_uncut = expectations(ref["p"], obs_spec) if ref["p"] is not None else None
    plans = enumerate_mixed(circ, g_max=g_max, w_max=w_max, max_width=max_width,
                            seed=sample_seed)
    by_type = {t: [p for p in plans if p.cut_type == t] for t in ("wire", "gate", "mixed")}
    order = by_type["wire"] + by_type["gate"]
    mixed = by_type["mixed"]
    order += [mixed[i] for i in rng.permutation(len(mixed))]
    target = len(order) if max_candidates is None else min(max_candidates, len(order))
    n_oracle = max(1, int(round(oracle_fraction * target)))
    oracle_types = set()
    records, n_too_wide = [], 0
    for plan in order:
        if len(records) >= target:
            break
        check = len(records) < n_oracle or plan.cut_type not in oracle_types
        try:
            rec = label_plan(ctx, circ, plan, p_ideal, f_uncut, check_oracle=check,
                             obs_spec=obs_spec, obs_ideal=obs_ideal)
        except MappedWidthExceeded:
            n_too_wide += 1
            continue
        if check:
            oracle_types.add(plan.cut_type)
        rec["circuit_meta"] = meta
        records.append(rec)
    return {"circuit_meta": meta, "f_uncut": None if f_uncut is None else float(f_uncut),
            "uncut_phys": ref["phys"], "uncut_n_2q_mapped": ref["n_2q_mapped"],
            "uncut_depth": ref["depth"], "uncut_active_qubits": ref["active_qubits"],
            "n_candidates_total": len(plans),
            "n_candidates_by_type": {t: len(v) for t, v in by_type.items()},
            "n_candidates_labelled": len(records),
            "n_oracle_checked": sum(1 for r in records if "oracle_tvd" in r),
            "n_excluded_mapped_width": n_too_wide, "g_max": g_max, "w_max": w_max,
            "max_width": max_width, "sample_seed": sample_seed, "label_scheme": "v3.1",
            "obs_spec": [list(S) for S in obs_spec],
            "obs_ideal": [float(x) for x in obs_ideal],
            "obs_uncut": None if obs_uncut is None else [float(x) for x in obs_uncut],
            "n_distinct_overhead": len({round(r["log2_sampling_overhead"], 9) for r in records}),
            "node_features": device_node_features(ctx, circ, ref["phys"]),
            "records": records}
