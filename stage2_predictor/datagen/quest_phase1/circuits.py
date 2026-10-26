"""Minimal, dependency-free quantum circuit representation.

Design note (QUEST-MPI Phase 1): this module is the *mathematical core* used to
validate the QPD combiner at machine precision in a sandbox without qiskit.
Production runs use qiskit/Aer via qiskit_bridge.py; the physics and the
contraction logic validated here are backend-independent.

Conventions
-----------
* Qubits are integers 0..n-1.
* State tensors have shape (2,)*n with axis q <-> qubit q.
* Flattened indices are big-endian in axis order: for axes [q0, q1, ...],
  index = z_{q0}*2^{m-1} + z_{q1}*2^{m-2} + ...  Consistency, not endianness,
  is what correctness depends on; all flattening goes through `flatten_axes`.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------- gate matrices
_SQ2 = 1.0 / math.sqrt(2.0)

_H = np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_T = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def _rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(t):
    return np.array([[cmath.exp(-1j * t / 2), 0], [0, cmath.exp(1j * t / 2)]], dtype=complex)


_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)  # |c t> basis
_CZ = np.diag([1, 1, 1, -1]).astype(complex)


def _rzz(t):
    e0, e1 = cmath.exp(-1j * t / 2), cmath.exp(1j * t / 2)
    return np.diag([e0, e1, e1, e0]).astype(complex)


_FIXED_1Q = {"h": _H, "x": _X, "y": _Y, "z": _Z, "s": _S, "t": _T, "id": _I2}
_PARAM_1Q = {"rx": _rx, "ry": _ry, "rz": _rz}
_FIXED_2Q = {"cx": _CX, "cz": _CZ}
_PARAM_2Q = {"rzz": _rzz}


# --------------------------------------------------------------------- circuit
@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple  # (q,) or (q1, q2); for cx: (control, target)
    params: tuple = ()

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    def matrix(self) -> np.ndarray:
        if self.name in _FIXED_1Q:
            return _FIXED_1Q[self.name]
        if self.name in _PARAM_1Q:
            return _PARAM_1Q[self.name](*self.params)
        if self.name in _FIXED_2Q:
            return _FIXED_2Q[self.name]
        if self.name in _PARAM_2Q:
            return _PARAM_2Q[self.name](*self.params)
        raise ValueError(f"unknown gate {self.name}")


@dataclass
class Circuit:
    n_qubits: int
    gates: list = field(default_factory=list)

    def add(self, name: str, qubits, *params) -> "Circuit":
        qs = (qubits,) if isinstance(qubits, int) else tuple(qubits)
        for q in qs:
            if not 0 <= q < self.n_qubits:
                raise ValueError(f"qubit {q} out of range")
        if len(set(qs)) != len(qs):
            raise ValueError("duplicate qubit in gate")
        self.gates.append(Gate(name, qs, tuple(params)))
        return self

    # DAG: edge u -> v iff v is the next gate after u on some shared qubit.
    def dag_edges(self) -> list:
        last: dict = {}
        edges = []
        for i, g in enumerate(self.gates):
            for q in g.qubits:
                if q in last:
                    edges.append((last[q], i, q))  # (u, v, wire qubit)
                last[q] = i
        return edges

    def predecessors(self) -> list:
        preds: list = [set() for _ in self.gates]
        for u, v, _q in self.dag_edges():
            preds[v].add(u)
        return preds

    def touched(self, gate_indices) -> set:
        qs: set = set()
        for i in gate_indices:
            qs.update(self.gates[i].qubits)
        return qs

    def subcircuit(self, gate_indices, qubit_map: dict) -> "Circuit":
        """Gates (in original temporal order) remapped onto a local register."""
        sub = Circuit(len(qubit_map))
        for i in sorted(gate_indices):
            g = self.gates[i]
            sub.gates.append(Gate(g.name, tuple(qubit_map[q] for q in g.qubits), g.params))
        return sub


# ------------------------------------------------------------------- utilities
def flatten_axes(tensor: np.ndarray, axes: Sequence[int]) -> np.ndarray:
    """Transpose `axes` to the front (given order) and flatten them into one
    big-endian index; remaining axes keep their relative order after it."""
    rest = [a for a in range(tensor.ndim) if a not in axes]
    t = np.transpose(tensor, list(axes) + rest)
    return t.reshape((2 ** len(axes),) + t.shape[len(axes):])
