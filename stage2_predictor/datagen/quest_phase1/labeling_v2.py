"""Labelling scheme v2: device-mapped fragments with a noisy cut instrument.

Why v2 exists (pilot finding, 10 Oct 2026)
------------------------------------------
Under the v1 scheme (gate-local Markovian noise, every logical qubit kept on
the same physical qubit, ideal cut instrument, infinite shots) the wire-cut
reconstruction equals the *noisy uncut* distribution exactly, for every cut:
the QPD identity is an exact linear decomposition of the identity channel and
noise channels act linearly, so

    sum_i c_i  Lambda_B( P_i( Lambda_A(rho) ) ) = Lambda_B( Lambda_A(rho) ).

All v1 labels of a circuit are therefore identical up to numerical error
(verified to 1e-15 once the v1 qubit-relabelling bug is removed), and they
carry no information about cut placement.

What v2 adds (the mechanisms through which cut placement changes fidelity)
------------------------------------------------------------------------
1. Independent device mapping of each fragment. Fragment A and fragment B
   are transpiled separately onto the device coupling map (fixed seed, fixed
   optimisation level). Layout, routing SWAPs and the physical qubits used
   depend on each fragment's interaction graph, so they depend on the cut.
   This is the mechanism measured by Ren, Zhang and Barbalace (arXiv:2409.03870).
2. Noisy cut instrument. The upstream measurement of each cut wire suffers
   thermal relaxation for the measurement duration plus an assignment error,
   modelled as a depolarizing channel with p = 2*eps (an assignment error eps
   shrinks every Pauli expectation by (1 - 2 eps), which is exactly the action
   of depolarizing noise with p = 2 eps). Downstream preparation uses real
   basis gates on the mapped physical qubit, so it carries gate noise.
3. Final readout error per *physical* qubit, applied classically after
   combination (symmetric confusion with the backend's assignment error).

Labels remain deterministic (density-matrix simulation, infinite-shot limit)
and reproducible from (circuit seed, backend snapshot, transpiler seed).
The sampling overhead of the QPD is NOT folded into the fidelity label; it is
recorded separately as log2(gamma^2) per cut so that Stage 3 can charge it
explicitly through the alpha term without double counting.

The noiseless path through the same mapped pipeline is the correctness
oracle: with noise, instrument error and readout error disabled, the
reconstruction must equal the ideal distribution to machine precision. It
checks the layout and qubit-order bookkeeping on every candidate.

Stated assumptions (thesis Ch. 5): fragments run as separate jobs on the same
device snapshot; a fresh qubit starts in |0> (no reset error modelled); idle
qubits during the cut measurement are not modelled beyond the instrument
channel; the calibration snapshot is the archived IBM fake backend shipped
with the pinned qiskit-ibm-runtime version.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .candidates import CutCandidate
from .circuits import Circuit
from .combiner import _FRAME, _TINV, _embed_full, fragment
from .simulators import hellinger_fidelity, probabilities, total_variation

LOG2_GAMMA2_WIRE = math.log2(16.0)  # LO wire cut without classical communication

# Frame-state preparation from |0>: |0>, |1>, |+>, |+i>  (matches combiner._FRAME)


# --------------------------------------------------------------- device context
@dataclass
class DeviceContext:
    backend: object
    seed: int = 1234
    opt_level: int = 2
    noisy: bool = True
    max_active_qubits: int = 13   # density-matrix memory ceiling AFTER routing
    name: str = ""
    calibration_date: str = ""
    readout_eps: dict = field(default_factory=dict)   # phys -> assignment error
    meas_duration: dict = field(default_factory=dict)  # phys -> seconds
    t1: dict = field(default_factory=dict)
    t2: dict = field(default_factory=dict)
    _sim: object = None

    def simulator(self):
        if self._sim is None:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel
            if self.noisy:
                nm = NoiseModel.from_backend(self.backend, readout_error=False)
                self._sim = AerSimulator(method="density_matrix", noise_model=nm)
            else:
                self._sim = AerSimulator(method="density_matrix")
        return self._sim

    def ideal_copy(self) -> "DeviceContext":
        return DeviceContext(self.backend, self.seed, self.opt_level, noisy=False,
                             max_active_qubits=self.max_active_qubits,
                             name=self.name, calibration_date=self.calibration_date,
                             readout_eps=self.readout_eps,
                             meas_duration=self.meas_duration, t1=self.t1, t2=self.t2)


def make_device(name: str = "fake_brisbane", seed: int = 1234,
                opt_level: int = 2) -> DeviceContext:
    """Archived IBM calibration snapshot (qiskit-ibm-runtime fake backend)."""
    from qiskit_ibm_runtime import fake_provider
    cls = {"fake_brisbane": fake_provider.FakeBrisbane,
           "fake_torino": fake_provider.FakeTorino,
           "fake_sherbrooke": fake_provider.FakeSherbrooke}[name]
    backend = cls()
    target = backend.target
    ctx = DeviceContext(backend, seed, opt_level, True, name=backend.name)
    props = backend.properties()
    ctx.calibration_date = str(props.last_update_date) if props is not None else ""
    for q in range(backend.num_qubits):
        m = target["measure"].get((q,))
        ctx.readout_eps[q] = float(m.error) if m is not None and m.error is not None else 0.0
        ctx.meas_duration[q] = float(m.duration) if m is not None and m.duration else 0.0
        qp = target.qubit_properties[q]
        ctx.t1[q] = float(qp.t1) if qp.t1 else float("inf")
        ctx.t2[q] = float(qp.t2) if qp.t2 else float("inf")
    return ctx


class MappedWidthExceeded(Exception):
    """Routing used more physical qubits than the simulation ceiling allows."""


# ----------------------------------------------------------- circuit utilities
def _to_qiskit(circ: Circuit):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(circ.n_qubits)
    for g in circ.gates:
        getattr(qc, g.name)(*g.params, *g.qubits)
    return qc


def _two_qubit_count(qc) -> int:
    return sum(1 for inst in qc.data
               if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")


@dataclass
class MappedRun:
    rho: np.ndarray          # (2,)*(2n), axis i <-> local qubit i (numpy convention)
    initial: list            # local qubit -> physical qubit at start
    final: list              # local qubit -> physical qubit at end
    n_2q_logical: int
    n_2q_mapped: int
    depth_mapped: int


def _transpile(ctx: DeviceContext, circ: Circuit, initial_layout: Optional[list] = None):
    from qiskit import transpile
    n = circ.n_qubits
    qc = _to_qiskit(circ)
    tqc = transpile(qc, backend=ctx.backend, initial_layout=initial_layout,
                    optimization_level=ctx.opt_level, seed_transpiler=ctx.seed)
    initial = list(tqc.layout.initial_index_layout(filter_ancillas=True))[:n]
    final = list(tqc.layout.final_index_layout(filter_ancillas=True))[:n]
    active = {tqc.find_bit(q).index for inst in tqc.data
              if inst.operation.name not in ("barrier", "delay") for q in inst.qubits}
    if len(active) > ctx.max_active_qubits:
        raise MappedWidthExceeded(f"{len(active)} active qubits after routing "
                                  f"(ceiling {ctx.max_active_qubits})")
    return qc, tqc, initial, final


# Frame preparation from |0> in native basis gates (valid on IBM sx/rz/x devices):
# |1> = X|0>;  |+> ~ RZ(pi/2) SX|0>;  |+i> ~ RZ(pi) SX|0>  (global phases dropped)
_PREP_NATIVE = {0: [], 1: [("x", None)], 2: [("sx", None), ("rz", math.pi / 2)],
                3: [("sx", None), ("rz", math.pi)]}


def _simulate(ctx: DeviceContext, tqc, n: int, final: list,
              prep_phys: Optional[dict] = None, instrument_qubits: tuple = ()) -> np.ndarray:
    """Density matrix of the transpiled circuit on the logical qubits (numpy
    convention). prep_phys: {physical qubit: frame index} prepended natively,
    so preparation carries the device's gate noise without re-transpiling."""
    from qiskit import QuantumCircuit
    from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error

    run = tqc.copy()
    if prep_phys:
        pre = QuantumCircuit(tqc.num_qubits)
        for pq, s in prep_phys.items():
            for name, theta in _PREP_NATIVE[s]:
                (pre.rz(theta, pq) if name == "rz" else getattr(pre, name)(pq))
        run = run.compose(pre, front=True)
    if ctx.noisy:
        for i in instrument_qubits:
            p = final[i]
            t1, t2 = ctx.t1[p], ctx.t2[p]
            err = depolarizing_error(min(1.0, 2.0 * ctx.readout_eps[p]), 1)
            if ctx.meas_duration[p] > 0 and math.isfinite(t1):
                err = thermal_relaxation_error(t1, min(t2, 2 * t1),
                                               ctx.meas_duration[p]).compose(err)
            run.append(err.to_instruction(), [p])
    run.save_density_matrix(qubits=[final[i] for i in range(n)])
    res = ctx.simulator().run(run).result()
    dm = np.asarray(res.data()["density_matrix"]).reshape((2,) * (2 * n))
    # qiskit: listed qubit 0 is least significant -> reverse to axis i <-> qubit i
    perm = list(reversed(range(n))) + [n + a for a in reversed(range(n))]
    return np.transpose(dm, perm)


def _run_mapped(ctx: DeviceContext, circ: Circuit, initial_layout: Optional[list] = None,
                instrument_qubits: tuple = ()) -> MappedRun:
    qc, tqc, initial, final = _transpile(ctx, circ, initial_layout)
    rho = _simulate(ctx, tqc, circ.n_qubits, final, instrument_qubits=instrument_qubits)
    return MappedRun(rho, initial, final, _two_qubit_count(qc),
                     _two_qubit_count(tqc), tqc.depth())


def _diag(rho: np.ndarray, n: int) -> np.ndarray:
    return np.real(np.einsum(rho.reshape(2 ** n, 2 ** n), [0, 0], [0])).reshape((2,) * n)


def _apply_confusion(p: np.ndarray, eps_by_axis: dict) -> np.ndarray:
    for ax, e in eps_by_axis.items():
        m = np.array([[1 - e, e], [e, 1 - e]])
        p = np.moveaxis(np.tensordot(m, p, axes=(1, ax)), 0, ax)
    return p


# ------------------------------------------------------------- reconstruction
def reconstruct_mapped(ctx: DeviceContext, circ: Circuit, cand: CutCandidate) -> dict:
    """Two-fragment wire-cut reconstruction with device-mapped fragments."""
    fr = fragment(circ, cand)
    nk, nc, nr = len(fr.k_qubits), len(fr.c_qubits), len(fr.r_qubits)
    na = nk + nc

    run_a = _run_mapped(ctx, fr.frag_a, instrument_qubits=tuple(range(nk, na)))
    rho_flat = run_a.rho.reshape(2 ** nk, 2 ** nc, 2 ** nk, 2 ** nc)
    m = np.einsum("iaib->iab", rho_flat)

    qc_b, tqc_b, layout_b, finals_b = _transpile(ctx, fr.frag_b)  # one mapping for all frames
    q_frames = np.empty((4,) * nc + (2 ** (nc + nr),), dtype=float)
    for s_flat in range(4 ** nc):
        s = [(s_flat // 4 ** (nc - 1 - j)) % 4 for j in range(nc)]
        prep = {layout_b[j]: s[j] for j in range(nc) if s[j] != 0}
        rho_b = _simulate(ctx, tqc_b, nc + nr, finals_b, prep_phys=prep)
        q_frames[tuple(s)] = _diag(rho_b, nc + nr).reshape(-1)

    w = q_frames.astype(complex)
    for j in range(nc):
        w = np.moveaxis(np.tensordot(_TINV.T, w, axes=(1, j)), 0, j)
    w = w.reshape((2, 2) * nc + (2 ** (nc + nr),))
    w = np.transpose(w, [2 * j for j in range(nc)] + [2 * j + 1 for j in range(nc)] + [2 * nc])
    w = w.reshape(2 ** nc, 2 ** nc, 2 ** (nc + nr))

    p = np.einsum("iab,abz->iz", m, w)
    if np.max(np.abs(p.imag)) > 1e-9:
        raise AssertionError("combiner imaginary residue")
    p = np.clip(p.real, 0.0, None)
    total = p.sum()
    if abs(total - 1.0) > 1e-6:
        raise AssertionError(f"combiner norm residue: {total}")
    p = (p / total).reshape((2,) * (nk + nc + nr))
    p_full = _embed_full(p, fr, circ.n_qubits)

    phys_final = {}
    for i, q in enumerate(fr.k_qubits):
        phys_final[q] = run_a.final[i]
    for i, q in enumerate(fr.c_qubits + fr.r_qubits):
        phys_final[q] = finals_b[i]
    if ctx.noisy:
        p_full = _apply_confusion(p_full, {q: ctx.readout_eps[pq]
                                           for q, pq in phys_final.items()})
    return {"p": p_full,
            "phys_a": [int(x) for x in run_a.initial],
            "phys_b": [int(x) for x in layout_b],
            "n_2q_logical": run_a.n_2q_logical + _two_qubit_count(qc_b),
            "n_2q_mapped": run_a.n_2q_mapped + _two_qubit_count(tqc_b),
            "depth_a": run_a.depth_mapped, "depth_b": tqc_b.depth(),
            "width_a": na, "width_b": nc + nr}


def uncut_reference(ctx: DeviceContext, circ: Circuit, simulate: bool = True) -> dict:
    """Uncut circuit on the device. The layout is always computed (it anchors the
    cut-independent node features); the noisy simulation only when feasible."""
    saved = ctx.max_active_qubits
    ctx.max_active_qubits = 10 ** 6            # layout only: no ceiling
    try:
        qc, tqc, initial, final = _transpile(ctx, circ)
    finally:
        ctx.max_active_qubits = saved
    active = {tqc.find_bit(q).index for inst in tqc.data
              if inst.operation.name not in ("barrier", "delay") for q in inst.qubits}
    simulate = simulate and len(active) <= ctx.max_active_qubits
    out = {"phys": [int(x) for x in initial], "n_2q_mapped": _two_qubit_count(tqc),
           "depth": tqc.depth(), "p": None, "active_qubits": len(active)}
    if simulate:
        p = _diag(_simulate(ctx, tqc, circ.n_qubits, final), circ.n_qubits)
        if ctx.noisy:
            p = _apply_confusion(p, {q: ctx.readout_eps[final[q]]
                                     for q in range(circ.n_qubits)})
        out["p"] = p
    return out


# -------------------------------------------------------------------- labels
def label_candidate_v2(ctx: DeviceContext, circ: Circuit, cand: CutCandidate,
                       p_ideal: Optional[np.ndarray] = None,
                       f_uncut: Optional[float] = None,
                       check_oracle: bool = True) -> dict:
    if p_ideal is None:
        p_ideal = probabilities(circ)
    t0 = time.perf_counter()
    out = {"k": cand.k, "cut_qubits": list(cand.cut_qubits),
           "cut_positions": [list(t) for t in cand.cut_positions],
           "cut_type": cand.cut_type, "n_gates_a": len(cand.part_a),
           "part_a": sorted(cand.part_a), "n_gates_total": len(circ.gates),
           "log2_sampling_overhead": LOG2_GAMMA2_WIRE * cand.k,
           "label_scheme": "v2"}
    if check_oracle:
        ideal = reconstruct_mapped(ctx.ideal_copy(), circ, cand)
        tvd = total_variation(ideal["p"], p_ideal)
        if tvd > 1e-8:
            raise AssertionError(f"mapped oracle violated: TVD={tvd:.2e}")
        out["oracle_tvd"] = float(tvd)
    rec = reconstruct_mapped(ctx, circ, cand)
    out["pst_hellinger"] = float(hellinger_fidelity(rec["p"], p_ideal))
    out["tv_distance"] = float(total_variation(rec["p"], p_ideal))
    if f_uncut is not None:
        out["f_uncut"] = float(f_uncut)
        out["delta_vs_uncut"] = out["pst_hellinger"] - float(f_uncut)
    for key in ("phys_a", "phys_b", "n_2q_logical", "n_2q_mapped",
                "depth_a", "depth_b", "width_a", "width_b"):
        out[key] = rec[key]
    out["routing_overhead_2q"] = rec["n_2q_mapped"] - rec["n_2q_logical"]
    out["label_wall_s"] = time.perf_counter() - t0
    out["device"] = ctx.name
    out["calibration_date"] = ctx.calibration_date
    out["transpiler_seed"] = ctx.seed
    out["opt_level"] = ctx.opt_level
    return out


def fragment_widths(circ: Circuit, cand: CutCandidate) -> tuple:
    fr = fragment(circ, cand)
    return len(fr.k_qubits) + len(fr.c_qubits), len(fr.c_qubits) + len(fr.r_qubits)


def label_circuit_v2(ctx: DeviceContext, circ: Circuit, meta: dict,
                     k_max: int = 2, check_oracle: bool = True,
                     max_fragment_width: int = 13, max_candidates: Optional[int] = None,
                     oracle_fraction: float = 1.0, sample_seed: int = 0,
                     uncut_max_qubits: int = 13) -> dict:
    """Label one circuit. Candidates whose larger fragment exceeds
    max_fragment_width are excluded (density-matrix memory ceiling); if more
    than max_candidates remain, a seeded uniform sample is labelled. The
    mapped oracle runs on a seeded oracle_fraction of candidates (always >= 1).
    Every exclusion and sampling decision is recorded in the output."""
    from .candidates import enumerate_candidates, validate_candidate
    rng = np.random.default_rng(sample_seed)
    p_ideal = probabilities(circ)
    simulate_uncut = circ.n_qubits <= uncut_max_qubits
    ref = uncut_reference(ctx, circ, simulate=simulate_uncut)
    f_uncut = hellinger_fidelity(ref["p"], p_ideal) if ref["p"] is not None else None

    all_cands = enumerate_candidates(circ, k_max=k_max)
    eligible = [c for c in all_cands if max(fragment_widths(circ, c)) <= max_fragment_width]
    order = rng.permutation(len(eligible)).tolist()
    target = len(eligible) if max_candidates is None else min(max_candidates, len(eligible))
    n_oracle = max(1, int(round(oracle_fraction * target))) if check_oracle else 0

    records, n_too_wide = [], 0
    for idx in order:
        if len(records) >= target:
            break
        cand = eligible[idx]
        validate_candidate(circ, cand)
        try:
            rec = label_candidate_v2(ctx, circ, cand, p_ideal, f_uncut,
                                     check_oracle=len(records) < n_oracle)
        except MappedWidthExceeded:
            n_too_wide += 1
            continue
        rec["circuit_meta"] = meta
        records.append(rec)
    chosen = records
    oracle_idx = [r for r in records if "oracle_tvd" in r]
    return {"circuit_meta": meta, "f_uncut": None if f_uncut is None else float(f_uncut),
            "uncut_phys": ref["phys"], "uncut_n_2q_mapped": ref["n_2q_mapped"],
            "uncut_depth": ref["depth"],
            "n_candidates_total": len(all_cands), "n_candidates_eligible": len(eligible),
            "n_candidates_labelled": len(chosen), "n_oracle_checked": len(oracle_idx),
            "n_excluded_mapped_width": n_too_wide, "uncut_active_qubits": ref["active_qubits"],
            "max_fragment_width": max_fragment_width, "sample_seed": sample_seed,
            "node_features": device_node_features(ctx, circ, ref["phys"]),
            "records": records}


def device_node_features(ctx: DeviceContext, circ: Circuit, phys: list) -> list:
    """Per-gate hardware features read from the physical qubits that the
    *uncut* circuit maps to (a cut-independent reference layout, so features
    do not leak the label). Both aggregation channels are kept (D3)."""
    target = ctx.backend.target
    two_q = [n for n in target.operation_names
             if n in ("ecr", "cz", "cx") and target.operation_from_name(n).num_qubits == 2]
    feats = []
    for g in circ.gates:
        pq = [phys[q] for q in g.qubits]
        t1s = [ctx.t1[p] * 1e6 for p in pq]
        t2s = [ctx.t2[p] * 1e6 for p in pq]
        if g.n_qubits == 1:
            props = target["sx"].get((pq[0],))
            err = float(props.error) if props is not None and props.error else 0.0
            coupled = 1.0
        else:
            errs = [target[n][pair].error for n in two_q
                    for pair in ((pq[0], pq[1]), (pq[1], pq[0]))
                    if pair in target[n] and target[n][pair].error is not None]
            coupled = 1.0 if errs else 0.0
            err = float(min(errs)) if errs else float(np.median(
                [p.error for n in two_q for p in target[n].values() if p.error]))
        feats.append({"gate_type": g.name,
                      "t1_min": min(t1s), "t1_mean": sum(t1s) / len(t1s),
                      "t2_min": min(t2s), "t2_mean": sum(t2s) / len(t2s),
                      "gate_error": err, "coupled_on_device": coupled,
                      "readout_eps_mean": float(np.mean([ctx.readout_eps[p] for p in pq])),
                      "_qubits_meta": list(g.qubits)})
    return feats
