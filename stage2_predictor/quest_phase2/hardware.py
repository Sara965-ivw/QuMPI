"""Hardware features from the physical qubits each FRAGMENT actually uses.

Why this module exists (RQ1 hardware ablation, 12 Oct 2026): the original
calibration features were read from the layout of the UNCUT circuit, while the
labels depend on where each fragment is placed. Removing those features changed
nothing (p = 0.63). The labeller stores each fragment's physical layout
(phys_a, phys_b), and that layout is known before execution (it comes from the
transpiler), so it is a legitimate model input.

Per candidate cut this module returns
  * per-gate calibration of the qubits the gate runs on IN ITS FRAGMENT:
    readout error, T1, T2 and gate error (4 channels);
  * a layout estimated success probability (layout-ESP) per fragment:
      log ESP = sum over 1q gates log(1 - e_sx(p))
              + sum over 2q gates log(1 - e_2q(p1, p2))
              + sum over cut measurements log(1 - 2 e_ro(p))
              + sum over output qubits log(1 - e_ro(p))
    where e_2q is the ECR error for coupled qubits and, for qubits that are not
    coupled, 1 - (1 - e_ecr_mean)^(1 + 3 (d - 1)) with d the coupling distance
    (a SWAP is three ECRs). This approximates the ESP of Nishio et al. (2020),
    which is computed on the compiled circuit; the approximation is stated
    wherever the ESP baseline is reported.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

N_LAYOUT = 4


@lru_cache(maxsize=1)
def _device():
    from quest_phase1.labeling_v2 import make_device
    ctx = make_device("fake_brisbane", seed=1234)
    tgt = ctx.backend.target
    e_sx, e_2q = {}, {}
    for (q,), props in tgt["sx"].items():
        e_sx[q] = props.error if props is not None and props.error is not None else 1e-3
    two_q = next(name for name in ("ecr", "cz", "cx") if name in tgt.operation_names)
    for qs, props in tgt[two_q].items():
        e = props.error if props is not None and props.error is not None else 1e-2
        e_2q[tuple(sorted(qs))] = min(e, e_2q.get(tuple(sorted(qs)), 1.0))
    e2_mean = float(np.mean([e for e in e_2q.values() if e < 0.5]))
    cmap = ctx.backend.coupling_map
    # typical (median) values: robust to the few broken qubits a noise-aware
    # layout avoids anyway
    fin = lambda d: float(np.median([v for v in d.values() if math.isfinite(v)]))
    means = {"sx": float(np.median(list(e_sx.values()))), "ro": fin(ctx.readout_eps),
             "t1": fin(ctx.t1), "t2": fin(ctx.t2),
             "e2": float(np.median([e for e in e_2q.values() if e < 0.5]))}
    return ctx, e_sx, e_2q, e2_mean, cmap, means


def _log1m(e: float) -> float:
    return math.log(max(1e-9, 1.0 - min(e, 0.999999)))


AGG_CHANNELS = {"mixed": 4, "min": 4, "mean": 4, "both": 7}


def layout_features(circ, rec: dict, plan, uniform: bool = False, agg: str = "mixed") -> dict:
    """plan: mixed_cuts.CutPlan of this record (k, c, r qubit order of the labeller).
    uniform=True is the calibration control: every qubit and coupler gets the
    device-median error, T1 and T2, while the placement and the coupling
    distances (hence the SWAP cost of uncoupled pairs) are kept. It separates
    'where the fragment sits on the chip' from 'how good those qubits are'."""
    ctx, e_sx, e_2q, e2_mean, cmap, means = _device()
    if uniform:
        class _U(dict):
            def __init__(self, v): super().__init__(); self.v = v
            def __getitem__(self, k): return self.v
            def __contains__(self, k): return False
        ro_map, t1_map, t2_map = _U(means["ro"]), _U(means["t1"]), _U(means["t2"])
        e_sx = _U(means["sx"]); e_2q = {}; e2_mean = means["e2"]
    else:
        ro_map, t1_map, t2_map = ctx.readout_eps, ctx.t1, ctx.t2
    pos_a = {q: i for i, q in enumerate(list(plan.k_qubits) + list(plan.c_qubits))}
    pos_b = {q: i for i, q in enumerate(list(plan.c_qubits) + list(plan.r_qubits))}
    phys_a, phys_b = rec["phys_a"], rec["phys_b"]
    cut = plan.cut
    cut_gates = set(plan.gate_cuts)

    def phys(q, side):
        return phys_a[pos_a[q]] if side == "A" else phys_b[pos_b[q]]

    def e2(p1, p2):
        key = tuple(sorted((p1, p2)))
        if key in e_2q:
            return e_2q[key]
        d = max(1, cmap.distance(p1, p2))
        return 1.0 - (1.0 - e2_mean) ** (1 + 3 * (d - 1))

    N = len(circ.gates)
    feats = np.zeros((N, AGG_CHANNELS[agg]), np.float32)
    log_esp = {"A": 0.0, "B": 0.0}
    for i, gt in enumerate(circ.gates):
        sides = [cut.side(q, i) for q in gt.qubits]
        ps = [phys(q, s) for q, s in zip(gt.qubits, sides)]
        ro = [ro_map[p] for p in ps]
        t1 = [t1_map[p] for p in ps]
        t2 = [t2_map[p] for p in ps]
        if i in cut_gates:                          # two local halves, one per fragment
            ge = float(np.mean([e_sx[p] for p in ps]))
            for p, s in zip(ps, sides):
                log_esp[s] += _log1m(e_sx[p]) + _log1m(2 * ro_map[p])
        elif gt.n_qubits == 2:
            ge = e2(ps[0], ps[1])
            log_esp[sides[0]] += _log1m(ge)
        else:
            ge = e_sx[ps[0]]
            log_esp[sides[0]] += _log1m(ge)
        # Aggregation over the qubits of a two-qubit gate (proposal lines 446-449, Card B):
        #   mixed: readout mean, T1 and T2 minimum (the frozen primary model; unchanged)
        #   min  : minimum of readout error, T1 and T2
        #   mean : mean of readout error, T1 and T2
        #   both : minimum and mean as separate columns
        # The gate error of a coupler is one value and is never aggregated.
        def _ro(v):
            return -math.log10(max(v, 1e-6)) / 3

        def _t(v):
            return math.log10(max(v, 1e-7)) / 3 if math.isfinite(v) else 0.0

        ge_c = -math.log10(max(ge, 1e-6)) / 4
        if agg == "mixed":
            feats[i] = [_ro(np.mean(ro)), _t(min(t1)), _t(min(t2)), ge_c]
        elif agg == "min":
            feats[i] = [_ro(min(ro)), _t(min(t1)), _t(min(t2)), ge_c]
        elif agg == "mean":
            feats[i] = [_ro(np.mean(ro)), _t(float(np.mean(t1))), _t(float(np.mean(t2))), ge_c]
        elif agg == "both":
            feats[i] = [_ro(min(ro)), _ro(np.mean(ro)), _t(min(t1)), _t(float(np.mean(t1))),
                        _t(min(t2)), _t(float(np.mean(t2))), ge_c]
        else:
            raise ValueError(agg)
    for q in cut.wire_qubits:                       # measure on A, prepare on B
        log_esp["A"] += _log1m(2 * ro_map[phys(q, "A")])
    for q in list(plan.k_qubits):
        log_esp["A"] += _log1m(ro_map[phys(q, "A")])
    for q in list(plan.c_qubits) + list(plan.r_qubits):
        log_esp["B"] += _log1m(ro_map[phys(q, "B")])
    t1a = [t1_map[p] for p in phys_a if math.isfinite(t1_map[p])]
    t1b = [t1_map[p] for p in phys_b if math.isfinite(t1_map[p])]
    return {"gate": feats, "log_esp_a": log_esp["A"], "log_esp_b": log_esp["B"],
            "t1_a": float(np.mean(t1a)) if t1a else 0.0, "t1_b": float(np.mean(t1b)) if t1b else 0.0}
