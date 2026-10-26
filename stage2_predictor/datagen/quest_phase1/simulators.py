"""Statevector and density-matrix simulators with a calibration-derived noise model.

The density-matrix path implements the noise physics of the amended D1 decision
(Phase 1 Kickoff Decision Record): per-gate depolarizing error followed by
per-qubit thermal relaxation (T1 amplitude damping + pure dephasing derived
from T2), plus classical readout confusion at measurement.

Noise application order per gate (documented assumption, mirrors Aer's default
composition): ideal unitary -> depolarizing(gate_error) on the gate's qubits ->
thermal relaxation(duration) on each involved qubit.

All arrays follow the tensor conventions of circuits.py. Density matrices have
shape (2,)*(2n): ket axes 0..n-1, bra axes n..2n-1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .circuits import Circuit, Gate

# ------------------------------------------------------------------ noise spec
@dataclass
class QubitNoise:
    t1_us: float
    t2_us: float
    readout_p01: float = 0.0  # P(read 1 | true 0)
    readout_p10: float = 0.0  # P(read 0 | true 1)

    def __post_init__(self):
        if self.t2_us > 2 * self.t1_us + 1e-9:
            raise ValueError("unphysical: T2 > 2*T1")


@dataclass
class NoiseSpec:
    """None => noiseless. gate_error_2q keys are frozenset({q1,q2})."""
    qubits: dict = field(default_factory=dict)          # q -> QubitNoise
    gate_error_1q: dict = field(default_factory=dict)   # q -> depolarizing prob
    gate_error_2q: dict = field(default_factory=dict)   # frozenset -> prob
    duration_1q_ns: float = 60.0
    duration_2q_ns: float = 500.0

    def relaxation_kraus(self, q: int, duration_ns: float):
        qn = self.qubits.get(q)
        if qn is None:
            return None
        t = duration_ns * 1e-3  # ns -> us
        g = 1.0 - math.exp(-t / qn.t1_us)                      # amplitude damping
        inv_tphi = 1.0 / qn.t2_us - 1.0 / (2.0 * qn.t1_us)     # pure dephasing rate
        p_phi = 0.5 * (1.0 - math.exp(-t * inv_tphi)) if inv_tphi > 0 else 0.0
        ad = [np.array([[1, 0], [0, math.sqrt(1 - g)]], dtype=complex),
              np.array([[0, math.sqrt(g)], [0, 0]], dtype=complex)]
        dz = [math.sqrt(1 - p_phi) * np.eye(2, dtype=complex),
              math.sqrt(p_phi) * np.diag([1, -1]).astype(complex)]
        return [d @ a for a in ad for d in dz]  # compose: AD then dephasing


_P1 = [np.eye(2, dtype=complex),
       np.array([[0, 1], [1, 0]], dtype=complex),
       np.array([[0, -1j], [1j, 0]], dtype=complex),
       np.diag([1, -1]).astype(complex)]


def depolarizing_kraus(p: float, n_qubits: int):
    if p <= 0:
        return None
    d = 4 ** n_qubits
    paulis = _P1 if n_qubits == 1 else [np.kron(a, b) for a in _P1 for b in _P1]
    ks = [math.sqrt(1 - p * (d - 1) / d) * paulis[0]]
    ks += [math.sqrt(p / d) * P for P in paulis[1:]]
    return ks


# ------------------------------------------------------------- statevector sim
def _apply_unitary(psi: np.ndarray, u: np.ndarray, axes: tuple) -> np.ndarray:
    k = len(axes)
    ut = u.reshape((2,) * (2 * k))
    psi = np.tensordot(ut, psi, axes=(list(range(k, 2 * k)), list(axes)))
    return np.moveaxis(psi, list(range(k)), list(axes))


def statevector(circ: Circuit, init: Optional[np.ndarray] = None) -> np.ndarray:
    n = circ.n_qubits
    if init is None:
        psi = np.zeros((2,) * n, dtype=complex)
        psi[(0,) * n] = 1.0
    else:
        psi = init.reshape((2,) * n).astype(complex)
    for g in circ.gates:
        psi = _apply_unitary(psi, g.matrix(), g.qubits)
    return psi


def product_state(states_1q: list) -> np.ndarray:
    """Tensor product of single-qubit state vectors, axis j <-> j-th entry."""
    psi = np.array([1.0], dtype=complex)
    for s in states_1q:
        psi = np.tensordot(psi, np.asarray(s, dtype=complex), axes=0)
    return psi[0] if psi.ndim > len(states_1q) else psi


# ---------------------------------------------------------- density-matrix sim
def _dm_apply_ops(rho: np.ndarray, ops: list, axes: tuple, n: int) -> np.ndarray:
    """rho -> sum_i K_i rho K_i^dagger on the given ket axes (bra: axes+n)."""
    bra_axes = tuple(a + n for a in axes)
    out = None
    for k_op in ops:
        term = _apply_unitary(rho, k_op, axes)
        term = _apply_unitary(term, k_op.conj(), bra_axes)
        out = term if out is None else out + term
    return out


def density_matrix(circ: Circuit, noise: Optional[NoiseSpec] = None,
                   init_psi: Optional[np.ndarray] = None) -> np.ndarray:
    n = circ.n_qubits
    psi = statevector(Circuit(n), init=init_psi) if init_psi is not None else None
    if psi is None:
        psi = np.zeros((2,) * n, dtype=complex)
        psi[(0,) * n] = 1.0
    rho = np.tensordot(psi, psi.conj(), axes=0)  # ket axes then bra axes
    for g in circ.gates:
        rho = _dm_apply_ops(rho, [g.matrix()], g.qubits, n)
        if noise is not None:
            perr = (noise.gate_error_1q.get(g.qubits[0], 0.0) if g.n_qubits == 1
                    else noise.gate_error_2q.get(frozenset(g.qubits), 0.0))
            dk = depolarizing_kraus(perr, g.n_qubits)
            if dk is not None:
                rho = _dm_apply_ops(rho, dk, g.qubits, n)
            dur = noise.duration_1q_ns if g.n_qubits == 1 else noise.duration_2q_ns
            for q in g.qubits:
                rk = noise.relaxation_kraus(q, dur)
                if rk is not None:
                    rho = _dm_apply_ops(rho, rk, (q,), n)
    return rho


def dm_diagonal(rho: np.ndarray, n: int) -> np.ndarray:
    """Diagonal of rho as a probability tensor of shape (2,)*n."""
    flat = rho.reshape(2 ** n, 2 ** n)
    return np.real(np.diagonal(flat)).reshape((2,) * n).copy()


def apply_readout(p: np.ndarray, noise: Optional[NoiseSpec], qubit_axes: dict) -> np.ndarray:
    """Classical readout confusion. qubit_axes maps physical qubit -> tensor axis."""
    if noise is None:
        return p
    for q, ax in qubit_axes.items():
        qn = noise.qubits.get(q)
        if qn is None or (qn.readout_p01 == 0 and qn.readout_p10 == 0):
            continue
        conf = np.array([[1 - qn.readout_p01, qn.readout_p10],
                         [qn.readout_p01, 1 - qn.readout_p10]])
        p = np.moveaxis(np.tensordot(conf, p, axes=(1, ax)), 0, ax)
    return p


# ---------------------------------------------------------------- distribution
def probabilities(circ: Circuit) -> np.ndarray:
    """Ideal (noiseless, infinite-shot) distribution, shape (2,)*n."""
    psi = statevector(circ)
    return np.abs(psi) ** 2


def hellinger_fidelity(p: np.ndarray, q: np.ndarray) -> float:
    bc = float(np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None))))
    return bc ** 2


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q)))
