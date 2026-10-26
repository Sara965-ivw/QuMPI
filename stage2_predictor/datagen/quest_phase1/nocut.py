"""The "no cut" option for separable circuits (decision A4, 22-23 Oct 2026).

Why this module exists
----------------------
Some generated circuits have no two-qubit gate between certain groups of
qubits: they already separate into independent pieces. If every piece has at
most 13 qubits, no cut is needed: the pieces run as separate jobs with no
quasiprobability decomposition and a sampling overhead of exactly 1.
qiskit-addon-cutting's find_cuts returns this option (zero cuts); the v3.1
candidate generator never does (plan_cut rejects a candidate without a cut).

Label (same pipeline and noise as every other label, scheme v3.1)
-----------------------------------------------------------------
Each piece is transpiled and simulated on its own on fake_brisbane, exactly as
the labeller treats the uncut circuit (uncut_reference: device noise model,
density matrix, readout confusion). Pieces are independent, so the joint
distribution is the tensor product of the piece distributions:

    p(x) = prod_j p_j(x restricted to the qubits of piece j).

There is no quasiprobability term, so the quasi-distribution equals p and the
sampling overhead gamma^2 = 1. The noiseless oracle checks that the product of
the ideal piece distributions reproduces the ideal distribution of the whole
circuit to machine precision.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .circuits import Circuit
from .labeling_v2 import DeviceContext, MappedWidthExceeded, uncut_reference
from .observables import basket, expectations
from .simulators import hellinger_fidelity, probabilities, total_variation


def pieces(circ: Circuit) -> list:
    """Connected components of the qubit interaction graph, each as a sorted qubit list."""
    parent = list(range(circ.n_qubits))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in circ.gates:
        if len(g.qubits) == 2:
            a, b = find(g.qubits[0]), find(g.qubits[1])
            if a != b:
                parent[a] = b
    groups: dict = {}
    for q in range(circ.n_qubits):
        groups.setdefault(find(q), []).append(q)
    return sorted((sorted(v) for v in groups.values()), key=lambda v: v[0])


def subcircuit(circ: Circuit, qubits: list) -> Circuit:
    """The gates acting on `qubits`, in their original order, on local indices 0..len-1."""
    loc = {q: i for i, q in enumerate(qubits)}
    sub = Circuit(len(qubits))
    for g in circ.gates:
        if g.qubits[0] in loc:
            if any(q not in loc for q in g.qubits):
                raise ValueError("a gate crosses pieces: the circuit is not separable here")
            sub.add(g.name, tuple(loc[q] for q in g.qubits), *g.params)
    return sub


def _tensor(dists: list, piece_qubits: list, n: int) -> np.ndarray:
    """Outer product of piece distributions, axes rearranged to global qubit order."""
    p = np.ones(())
    order = []
    for d, qs in zip(dists, piece_qubits):
        p = np.multiply.outer(p, np.asarray(d, float).reshape((2,) * len(qs)))
        order += qs
    return np.transpose(p, np.argsort(order)) if n else p


def label_nocut(ctx: DeviceContext, circ: Circuit, max_width: int = 13,
                p_ideal: Optional[np.ndarray] = None, obs_spec: Optional[list] = None,
                obs_ideal=None, check_oracle: bool = True) -> dict:
    """Label the no-cut option of a separable circuit. Raises ValueError if the circuit
    does not separate into pieces of at most `max_width` qubits, and
    MappedWidthExceeded if a piece needs more than the ceiling after routing."""
    t0 = time.perf_counter()
    parts = pieces(circ)
    if len(parts) < 2 or max(len(p) for p in parts) > max_width:
        raise ValueError(f"not separable into pieces of at most {max_width} qubits: "
                         f"{[len(p) for p in parts]}")
    if p_ideal is None:
        p_ideal = probabilities(circ)
    if obs_spec is None:
        obs_spec = basket(circ)
    if obs_ideal is None:
        obs_ideal = expectations(p_ideal, obs_spec)
    subs = [subcircuit(circ, qs) for qs in parts]
    out = {"cut_type": "none", "g": 0, "w": 0, "k": 0, "pieces": parts,
           "piece_widths": [len(p) for p in parts], "home_a": [], "wire_cuts": [],
           "gate_cut_indices": [], "width_a": max(len(p) for p in parts), "width_b": 0,
           "log2_sampling_overhead": 0.0, "gamma2": 1.0, "label_scheme": "v3.1+nocut"}
    if check_oracle:
        p_prod = _tensor([probabilities(s) for s in subs], parts, circ.n_qubits)
        tvd = total_variation(p_prod, p_ideal)
        if tvd > 1e-8:
            raise AssertionError(f"no-cut oracle violated: TVD={tvd:.2e}")
        out["oracle_tvd"] = float(tvd)
    refs = [uncut_reference(ctx, s, simulate=True) for s in subs]
    for s, r in zip(subs, refs):
        if r["p"] is None:
            raise MappedWidthExceeded(f"piece of {s.n_qubits} qubits needs {r['active_qubits']} "
                                      f"active qubits after routing (ceiling {ctx.max_active_qubits})")
    p = _tensor([r["p"] for r in refs], parts, circ.n_qubits)
    out["obs_mu"] = [float(x) for x in expectations(p, obs_spec)]
    out["pst_hellinger"] = float(hellinger_fidelity(p, p_ideal))
    out["tv_distance"] = float(total_variation(p, p_ideal))
    out["piece_phys"] = [r["phys"] for r in refs]
    out["piece_active_qubits"] = [r["active_qubits"] for r in refs]
    out["n_2q_mapped"] = int(sum(r["n_2q_mapped"] for r in refs))
    out["n_2q_logical"] = int(sum(1 for g in circ.gates if len(g.qubits) == 2))
    out["routing_overhead_2q"] = out["n_2q_mapped"] - out["n_2q_logical"]
    out["depth_pieces"] = [r["depth"] for r in refs]
    out["label_wall_s"] = time.perf_counter() - t0
    out.update(device=ctx.name, calibration_date=ctx.calibration_date,
               transpiler_seed=ctx.seed, opt_level=ctx.opt_level)
    return out
