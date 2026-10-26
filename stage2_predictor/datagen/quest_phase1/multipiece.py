"""Gate-cut reconstruction with any number of independent pieces (fix B, 23 Oct 2026).

Why this module exists
----------------------
mixed_cuts.reconstruct_plan has two sides, A and B. qiskit-addon-cutting's
find_cuts may split a circuit into three or more subcircuits joined only by gate
cuts. Putting several subcircuits on one side makes that side wider than any
single subcircuit, and the labeller rejected cuts that find_cuts itself had
made to fit. Here every subcircuit ("piece") is transpiled and simulated on its
own, which is how such a cut is executed: each subcircuit is a separate job.

Construction (identical building blocks to reconstruct_plan)
-----------------------------------------------------------
Pieces are the connected components of the qubit interaction graph after the
cut gates are removed. Each cut gate i is reduced to RZZ(theta_i) by the local
frame of _local_rzz_frame and decomposed exactly into the six terms of
_RZZ_TERMS (Mitarai and Fujii 2021), with weights _term_weights(theta_i). For a
joint term tau the full outcome distribution factorises over pieces:

    p = sum_tau  W_tau  prod_P  p_P(tau),      W_tau = prod_i w_{i, tau_i},

where p_P(tau) is the (signed, for "M" slots) outcome distribution of piece P
with the local operations of tau. Each piece uses one transpiled template per
pattern of measurement slots, with RZ slots bound per term and the layout of its
first template reused, exactly as reconstruct_plan does per side. Readout
confusion is a tensor product of single-qubit maps and is applied per piece.
The unclipped sum is the quasi-distribution (observables); the label uses the
standard projection (clip negatives, renormalise). With the noiseless context
the result must equal the ideal distribution (oracle).

Gate cuts only: a find_cuts result containing a wire cut is handled by the
two-side labeller (mixed_cuts) as before.
"""
from __future__ import annotations

import itertools
import math
import time
from typing import Optional

import numpy as np

from .circuits import Circuit
from .labeling_v2 import DeviceContext, MappedWidthExceeded, _apply_confusion
from .mixed_cuts import (CUTTABLE, _RZZ_TERMS, _SLOT_ANGLE, Mapped, _diag, _local_rzz_frame,
                         _term_weights, gate_cut_gamma, map_fragment, simulate_mapped)
from .observables import basket, expectations
from .simulators import hellinger_fidelity, probabilities, total_variation


def gate_cut_pieces(circ: Circuit, gate_cuts) -> list:
    """Connected components of the qubit interaction graph without the cut gates."""
    cut = set(gate_cuts)
    parent = list(range(circ.n_qubits))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, g in enumerate(circ.gates):
        if len(g.qubits) == 2 and i not in cut:
            a, b = find(g.qubits[0]), find(g.qubits[1])
            if a != b:
                parent[a] = b
    groups: dict = {}
    for q in range(circ.n_qubits):
        groups.setdefault(find(q), []).append(q)
    return sorted((sorted(v) for v in groups.values()), key=lambda v: v[0])


class _Structure:
    def __init__(self, circ: Circuit, gate_cuts):
        self.circ = circ
        self.cuts = sorted(gate_cuts)
        for i in self.cuts:
            if circ.gates[i].name not in CUTTABLE or len(circ.gates[i].qubits) != 2:
                raise ValueError(f"gate {i} ({circ.gates[i].name}) is not cuttable")
        self.pieces = gate_cut_pieces(circ, self.cuts)
        self.piece_of = {q: k for k, qs in enumerate(self.pieces) for q in qs}
        self.local = {q: qs.index(q) for qs in self.pieces for q in qs}
        for i in self.cuts:
            a, b = circ.gates[i].qubits
            if self.piece_of[a] == self.piece_of[b]:
                raise ValueError(f"cut gate {i} lies inside one piece: not a separating cut")
        # role "A" (first op of each _RZZ_TERMS entry) = the gate's first qubit. RZZ is
        # symmetric and the six terms contain both orientations, so any fixed role
        # assignment is an exact decomposition.
        self.slots = {k: [] for k in range(len(self.pieces))}       # piece -> [(cut, role)]
        for i in self.cuts:
            a, b = circ.gates[i].qubits
            self.slots[self.piece_of[a]].append((i, 0))
            self.slots[self.piece_of[b]].append((i, 1))

    def ops_for(self, k: int, ops: dict, params: Optional[dict]) -> list:
        """Local operation list of piece k given {cut: (op_role0, op_role1)}."""
        circ, frag = self.circ, []
        qs = set(self.pieces[k])

        def emit(q, name, p):
            if q in qs:
                frag.append((name, (self.local[q],), () if p is None else (p,)))

        for i, gt in enumerate(circ.gates):
            if i not in ops:
                if gt.qubits[0] in qs:
                    frag.append((gt.name, tuple(self.local[q] for q in gt.qubits), gt.params))
                continue
            theta, pre, post = _local_rzz_frame(gt)
            for q, name, p in pre:
                emit(q, name, p)
            for role, q in enumerate(gt.qubits):
                if q not in qs:
                    continue
                op = ops[i][role]
                if op == "M":
                    emit(q, "mz", None)
                elif params is not None:
                    emit(q, "rz", params[(i, role)])
                elif op != "I":
                    emit(q, "rz", _SLOT_ANGLE[op])
            for q, name, p in post:
                emit(q, name, p)
        return frag


def _terms(st: _Structure):
    weights = {i: _term_weights(math.pi / 2 if st.circ.gates[i].name in ("cx", "cz")
                                else float(st.circ.gates[i].params[0])) for i in st.cuts}
    for tau in itertools.product(range(6), repeat=len(st.cuts)):
        w = 1.0
        for i, t in zip(st.cuts, tau):
            w *= weights[i][t]
        if w == 0.0:
            continue
        yield w, {i: (_RZZ_TERMS[t][1], _RZZ_TERMS[t][2]) for i, t in zip(st.cuts, tau)}


def multipiece_fits(ctx: DeviceContext, circ: Circuit, gate_cuts) -> tuple:
    """Every template of every piece within the ceiling after routing (the same
    templates, order and layout reuse as the reconstruction)? Returns (fits, details)."""
    from qiskit.circuit import Parameter
    st = _Structure(circ, gate_cuts)
    params = {(i, r): Parameter(f"g{i}r{r}") for i in st.cuts for r in (0, 1)}
    done = {k: set() for k in range(len(st.pieces))}
    layout = {k: None for k in range(len(st.pieces))}
    result = {k: "ok" for k in range(len(st.pieces))}
    for _, ops in _terms(st):
        for k in range(len(st.pieces)):
            key = tuple(ops[i][r] == "M" for i, r in st.slots[k])
            if key in done[k] or result[k] != "ok":
                continue
            done[k].add(key)
            try:
                mp = map_fragment(ctx, len(st.pieces[k]), st.ops_for(k, ops, params),
                                  initial_layout=layout[k])
                if layout[k] is None:
                    layout[k] = mp.initial
            except MappedWidthExceeded as e:
                result[k] = str(e).split(" qubits")[0]
    det = [result[k] for k in range(len(st.pieces))]
    return all(v == "ok" for v in det), det


def reconstruct_multipiece(ctx: DeviceContext, circ: Circuit, gate_cuts) -> dict:
    from qiskit.circuit import Parameter
    st = _Structure(circ, gate_cuts)
    n, npc = circ.n_qubits, len(st.pieces)
    params = {(i, r): Parameter(f"g{i}r{r}") for i in st.cuts for r in (0, 1)}
    tmpl = {k: {} for k in range(npc)}
    layout = {k: None for k in range(npc)}
    cache = {k: {} for k in range(npc)}
    p = np.zeros((2,) * n)
    order = [q for qs in st.pieces for q in qs]
    perm = np.argsort(order)
    stats = {"n_2q_logical": 0, "n_2q_mapped": 0, "transpiles": 0, "runs": 0}
    final = {}
    for w, ops in _terms(st):
        dists = []
        for k in range(npc):
            slot_ops = tuple(ops[i][r] for i, r in st.slots[k])
            struct = tuple(o == "M" for o in slot_ops)
            if struct not in tmpl[k]:
                tmpl[k][struct] = map_fragment(ctx, len(st.pieces[k]), st.ops_for(k, ops, params),
                                               initial_layout=layout[k])
                stats["transpiles"] += 1
                if layout[k] is None:
                    m0 = tmpl[k][struct]
                    layout[k] = m0.initial
                    stats["n_2q_logical"] += m0.n_2q_logical
                    stats["n_2q_mapped"] += m0.n_2q_mapped
            if slot_ops not in cache[k]:
                m = tmpl[k][struct]
                values = {params[(i, r)]: _SLOT_ANGLE[o]
                          for (i, r), o in zip(st.slots[k], slot_ops) if o != "M"}
                live = {a: v for a, v in values.items() if a in m.tqc.parameters}
                tqc = m.tqc.assign_parameters(live) if live else m.tqc
                mb = Mapped(tqc, m.initial, m.final, m.n_meas, m.n_2q_logical, m.n_2q_mapped, m.depth)
                nl = len(st.pieces[k])
                d = _diag(simulate_mapped(ctx, mb, nl), nl)
                if ctx.noisy:      # readout confusion: per piece, linear, commutes with the sum
                    d = _apply_confusion(d, {j: ctx.readout_eps[m.final[j]] for j in range(nl)})
                cache[k][slot_ops] = d
                final[k] = m.final
                stats["runs"] += 1
            dists.append(cache[k][slot_ops])
        term = np.ones(())
        for d in dists:
            term = np.multiply.outer(term, d)
        p += w * np.transpose(term, perm)
    total = float(p.sum())
    neg_mass = float(-p[p < 0].sum())
    tol = 0.05 if ctx.noisy else 1e-6
    if abs(total - 1.0) > tol:
        raise AssertionError(f"multipiece combiner norm residue: {total}")
    p_quasi = p.copy()
    pc = np.clip(p, 0.0, None)
    return {"p": pc / pc.sum(), "p_quasi": p_quasi, "pieces": st.pieces,
            "piece_phys": [[int(x) for x in layout[k]] for k in range(npc)],
            "norm_residue": total - 1.0, "negative_mass": neg_mass, **stats}


def label_multipiece(ctx: DeviceContext, circ: Circuit, gate_cuts, p_ideal=None,
                     obs_spec: Optional[list] = None, obs_ideal=None,
                     check_oracle: bool = True) -> dict:
    t0 = time.perf_counter()
    if p_ideal is None:
        p_ideal = probabilities(circ)
    if obs_spec is None:
        obs_spec = basket(circ)
    if obs_ideal is None:
        obs_ideal = expectations(p_ideal, obs_spec)
    cuts = sorted(gate_cuts)
    log2 = sum(2 * math.log2(gate_cut_gamma(circ.gates[i].name, circ.gates[i].params)) for i in cuts)
    out = {"cut_type": "gate", "g": len(cuts), "w": 0, "k": len(cuts), "gate_cut_indices": cuts,
           "wire_cuts": [], "home_a": [], "log2_sampling_overhead": log2, "gamma2": float(2.0 ** log2),
           "label_scheme": "v3.1+multipiece"}
    if check_oracle:
        ideal = reconstruct_multipiece(ctx.ideal_copy(), circ, cuts)
        tvd = total_variation(ideal["p"], p_ideal)
        if tvd > 1e-8:
            raise AssertionError(f"multipiece oracle violated: TVD={tvd:.2e}")
        dmu = float(np.max(np.abs(expectations(ideal["p_quasi"], obs_spec) - obs_ideal)))
        if dmu > 1e-8:
            raise AssertionError(f"multipiece observable oracle violated: {dmu:.2e}")
        out["oracle_tvd"], out["oracle_obs_maxerr"] = float(tvd), dmu
    rec = reconstruct_multipiece(ctx, circ, cuts)
    out["pieces"] = rec["pieces"]
    out["piece_widths"] = [len(x) for x in rec["pieces"]]
    out["width_a"], out["width_b"] = max(out["piece_widths"]), 0
    out["obs_mu"] = [float(x) for x in expectations(rec["p_quasi"], obs_spec)]
    out["pst_hellinger"] = float(hellinger_fidelity(rec["p"], p_ideal))
    out["tv_distance"] = float(total_variation(rec["p"], p_ideal))
    for key in ("piece_phys", "norm_residue", "negative_mass", "n_2q_logical", "n_2q_mapped", "runs"):
        out[key] = rec[key]
    out["routing_overhead_2q"] = rec["n_2q_mapped"] - rec["n_2q_logical"]
    out["label_wall_s"] = time.perf_counter() - t0
    out.update(device=ctx.name, calibration_date=ctx.calibration_date,
               transpiler_seed=ctx.seed, opt_level=ctx.opt_level)
    return out
