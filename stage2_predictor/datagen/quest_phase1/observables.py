"""Fixed observable basket for finite-shot error analysis (label scheme v3.1).

Why
---
Fidelity labels describe the whole output distribution. The error a user
actually incurs is the error of an estimated expectation value at a finite
shot budget N. For a Pauli-Z string O_S (values +-1) estimated by the
quasi-probability (QPD) estimator of a cut with overhead gamma, every sample
takes the value +-gamma, so

    E[estimate]  = mu_k                       (exact expectation of the noisy,
                                               unclipped quasi-distribution)
    Var[1 shot]  = gamma_k^2 - mu_k^2         (exact, not a bound)
    MSE_k(N)     = (mu_k - mu_ideal)^2 + (gamma_k^2 - mu_k^2) / N.

Storing mu_ideal once per circuit and mu_k per candidate therefore gives the
exact finite-shot MSE of every candidate at every budget, with no further
simulation (Mitarai and Fujii 2021; Piveteau and Sutter 2024). The uncut
circuit is the special case gamma = 1.

The basket (fixed before holdout3 labelling; see PREREG_holdout3.md):
  * Z_i for every qubit i                        (local populations)
  * Z_i Z_j for every pair that shares a two-qubit gate
    (edges of the Qubit Interaction Graph; for QAOA these are the cost terms)
  * Z_0 Z_1 ... Z_{n-1}                          (global parity, most sensitive)

Convention: distributions have shape (2,)*n with axis i = logical qubit i,
the convention of simulators.probabilities and mixed_cuts.reconstruct_plan.
"""
from __future__ import annotations

import numpy as np

from .circuits import Circuit


def basket(circ: Circuit) -> list:
    n = circ.n_qubits
    singles = [(i,) for i in range(n)]
    pairs = sorted({tuple(sorted(g.qubits)) for g in circ.gates if len(g.qubits) == 2})
    spec = singles + pairs
    if n > 2:
        spec.append(tuple(range(n)))
    return spec


def expectations(p: np.ndarray, spec: list) -> np.ndarray:
    """<Z_S> = sum_x p(x) (-1)^{sum_{i in S} x_i}; p may be a quasi-distribution."""
    p = np.asarray(p, dtype=float)
    n = p.ndim
    out = np.empty(len(spec))
    sign = np.array([1.0, -1.0])
    for k, S in enumerate(spec):
        others = tuple(ax for ax in range(n) if ax not in S)
        marg = p.sum(axis=others) if others else p          # axes in increasing order of S
        par = np.ones((1,) * 0)
        for _ in S:
            par = np.multiply.outer(par, sign)
        out[k] = float(np.sum(marg * par))
    return out


def mse(mu: np.ndarray, mu_ideal: np.ndarray, gamma2: float, shots) -> np.ndarray:
    """Exact finite-shot MSE per observable; shots may be an array (broadcast)."""
    mu, mu_ideal = np.asarray(mu, float), np.asarray(mu_ideal, float)
    shots = np.asarray(shots, float)[..., None]
    return (mu - mu_ideal) ** 2 + (gamma2 - mu ** 2) / shots
