"""Exact QPD wire-cut combiner: noiseless oracle + noisy density-matrix labeler.

Mathematical structure (amended D1, Kickoff Decision Record):

Noiseless oracle (correctness test, must return fidelity 1 to machine precision):
    |psi_full> = U_B (|psi_A>_{K,C} (x) |0>_R)
    amp[zK, zB] = sum_a Psi[zK, a] * V[a, zB]
i.e. an amplitude-level contraction with bond dimension 2^k over the cut basis.
If this does not reproduce the ideal distribution exactly, the fragmenting or
index bookkeeping is wrong — it is the oracle for everything else.

Noisy labeler (produces the dataset labels):
    p[zK, zB] = sum_{a,b} M[zK, a, b] * W[a, b, zB]
    M[zK, a, b] = <zK, a| rho_A |zK, b>          (noisy upstream fragment)
    W[a, b, zB] = <zB| Lambda_B(|a><b|_C (x) |0><0|_R) |zB>   (noisy downstream channel)
W is obtained by linearity from 4^k simulations of the downstream fragment with
cut qubits prepared in the frame {|0>,|1>,|+>,|+i>} (a basis of 2x2 operator
space), then a per-qubit change of basis to |a><b| elements. This is the
"4^k branch simulations" of the cost model.

Assumption (v1, stated): ideal cut instrument — no noise on the virtual
measure/prepare at the frontier itself; gate + readout noise inside fragments.

Off-diagonal zK terms of rho_A are irrelevant because K qubits are measured in
the computational basis and U_B acts trivially on them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .candidates import CutCandidate
from .circuits import Circuit
from .simulators import (NoiseSpec, apply_readout, density_matrix, dm_diagonal,
                         hellinger_fidelity, probabilities, product_state,
                         statevector, total_variation)

_SQ2 = 1.0 / np.sqrt(2.0)
_FRAME = [np.array([1, 0], dtype=complex),            # |0>
          np.array([0, 1], dtype=complex),            # |1>
          np.array([_SQ2, _SQ2], dtype=complex),      # |+>
          np.array([_SQ2, 1j * _SQ2], dtype=complex)]  # |+i>


def _frame_inverse() -> np.ndarray:
    """Tinv[s, v] with v = 2*a + b:  |a><b| = sum_s Tinv[s, v] |F_s><F_s|."""
    s_mat = np.array([np.outer(f, f.conj()).reshape(4) for f in _FRAME])  # [s, v]
    return np.linalg.inv(s_mat.T)  # solves S^T c = e_v for every v


_TINV = _frame_inverse()


@dataclass
class Fragmentation:
    k_qubits: list   # upstream-only (measured from fragment A)
    c_qubits: list   # cut qubits (sorted)
    r_qubits: list   # downstream-only
    idle_qubits: list
    frag_a: Circuit  # register order: K + C
    frag_b: Circuit  # register order: C + R


def fragment(circ: Circuit, cand: CutCandidate) -> Fragmentation:
    b = set(range(len(circ.gates))) - cand.part_a
    qa, qb = circ.touched(cand.part_a), circ.touched(b)
    c = sorted(qa & qb)
    k = sorted(qa - qb)
    r = sorted(qb - qa)
    idle = sorted(set(range(circ.n_qubits)) - qa - qb)
    map_a = {q: i for i, q in enumerate(k + c)}
    map_b = {q: i for i, q in enumerate(c + r)}
    return Fragmentation(k, c, r, idle,
                         circ.subcircuit(cand.part_a, map_a),
                         circ.subcircuit(b, map_b))


def _embed_full(p_kcr: np.ndarray, fr: Fragmentation, n: int) -> np.ndarray:
    """Distribution over axes ordered K+C+R -> full-register axes 0..n-1
    (idle qubits deterministically 0)."""
    for _ in fr.idle_qubits:
        p_kcr = np.tensordot(p_kcr, np.array([1.0, 0.0]), axes=0)
    order = fr.k_qubits + fr.c_qubits + fr.r_qubits + fr.idle_qubits
    perm = [order.index(q) for q in range(n)]
    return np.transpose(p_kcr, perm)


# ------------------------------------------------------------ noiseless oracle
def reconstruct_noiseless(circ: Circuit, cand: CutCandidate) -> np.ndarray:
    fr = fragment(circ, cand)
    nk, nc, nr = len(fr.k_qubits), len(fr.c_qubits), len(fr.r_qubits)
    psi_a = statevector(fr.frag_a)                       # axes: K then C (local)
    # Psi[zK, a]: local axes are already ordered K then C by construction.
    psi_flat = psi_a.reshape(2 ** nk, 2 ** nc)
    # V[a, zB]: downstream statevector for each cut-basis input.
    v = np.empty((2 ** nc, 2 ** (nc + nr)), dtype=complex)
    for a in range(2 ** nc):
        bits = [(a >> (nc - 1 - j)) & 1 for j in range(nc)]
        init = product_state([_FRAME[bit] for bit in bits] + [_FRAME[0]] * nr)
        v[a] = statevector(fr.frag_b, init=init).reshape(-1)
    amp = psi_flat @ v                                   # [zK, zB]
    p = (np.abs(amp) ** 2).reshape((2,) * (nk + nc + nr))
    return _embed_full(p, fr, circ.n_qubits)


# --------------------------------------------------------------- noisy labeler
def reconstruct_noisy(circ: Circuit, cand: CutCandidate,
                      noise: Optional[NoiseSpec],
                      dm_func=None) -> np.ndarray:
    """dm_func(circuit, noise=..., init_psi=...) -> density-matrix tensor.
    Defaults to the validated numpy engine; qiskit_bridge supplies an
    Aer-backed drop-in for GPU-scale runs (cross-validated in test_bridge)."""
    if dm_func is None:
        dm_func = density_matrix
    fr = fragment(circ, cand)
    nk, nc, nr = len(fr.k_qubits), len(fr.c_qubits), len(fr.r_qubits)
    na = nk + nc

    # Upstream: M[zK, a, b] = <zK,a| rho_A |zK,b>  (diagonal in zK).
    rho_a = dm_func(fr.frag_a, noise=noise)              # (2,)*(2*na)
    rho_flat = rho_a.reshape(2 ** nk, 2 ** nc, 2 ** nk, 2 ** nc)
    m = np.einsum("iaib->iab", rho_flat)

    # Downstream: 4^k frame-input noisy simulations.
    q_frames = np.empty((4,) * nc + (2 ** (nc + nr),), dtype=float)
    for s_flat in range(4 ** nc):
        s = [(s_flat // 4 ** (nc - 1 - j)) % 4 for j in range(nc)]
        init = product_state([_FRAME[sj] for sj in s] + [_FRAME[0]] * nr)
        rho_b = dm_func(fr.frag_b, noise=noise, init_psi=init)
        q_frames[tuple(s)] = dm_diagonal(rho_b, nc + nr).reshape(-1)
    # Change of basis per cut qubit: frame index s_j -> operator element (a_j, b_j).
    w = q_frames.astype(complex)
    for j in range(nc):
        w = np.moveaxis(np.tensordot(_TINV.T, w, axes=(1, j)), 0, j)
    # Axes now (v_0..v_{nc-1}, zB) with v = 2a + b; split and group a's, b's.
    w = w.reshape((2, 2) * nc + (2 ** (nc + nr),))
    a_axes = [2 * j for j in range(nc)]
    b_axes = [2 * j + 1 for j in range(nc)]
    w = np.transpose(w, a_axes + b_axes + [2 * nc])
    w = w.reshape(2 ** nc, 2 ** nc, 2 ** (nc + nr))       # W[a, b, zB]

    p = np.einsum("iab,abz->iz", m, w)
    assert np.max(np.abs(p.imag)) < 1e-9, "combiner imaginary residue"
    p = np.clip(p.real, 0.0, None)
    total = p.sum()
    assert abs(total - 1.0) < 1e-6, f"combiner norm residue: {total}"
    p = (p / total).reshape((2,) * (nk + nc + nr))
    p_full = _embed_full(p, fr, circ.n_qubits)
    # Readout confusion on all measured qubits (cut instrument itself ideal, v1).
    return apply_readout(p_full, noise, {q: q for q in range(circ.n_qubits)})


# -------------------------------------------------------------------- labeling
def label_candidate(circ: Circuit, cand: CutCandidate,
                    noise: Optional[NoiseSpec],
                    check_oracle: bool = True, dm_func=None) -> dict:
    p_ideal = probabilities(circ)
    result = {"k": cand.k, "cut_qubits": list(cand.cut_qubits),
              "cut_positions": [list(t) for t in cand.cut_positions],
              "cut_type": cand.cut_type, "n_gates_a": len(cand.part_a),
              "n_gates_total": len(circ.gates)}
    if check_oracle:  # standing invariant: exact path must be exact
        p_oracle = reconstruct_noiseless(circ, cand)
        f_oracle = hellinger_fidelity(p_oracle, p_ideal)
        assert abs(f_oracle - 1.0) < 1e-9, f"oracle broken: F={f_oracle}"
        result["oracle_fidelity"] = f_oracle
    p_noisy = reconstruct_noisy(circ, cand, noise, dm_func=dm_func)
    result["pst_hellinger"] = hellinger_fidelity(p_noisy, p_ideal)
    result["tv_distance"] = total_variation(p_noisy, p_ideal)
    return result
