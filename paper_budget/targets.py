"""Bias and per-shot variance labels for every candidate cut (scheme v3.1).

Definition identical to stage2_predictor/analysis/budget_h3.py (pre-registered):
    MSE_k(N) = b2_k + v_k / N
    b2_k = mean_S (mu_kS - mu*_S)^2          (squared bias, basket mean)
    v_k  = mean_S max(gamma_k^2 - mu_kS^2, 0) (per-shot variance, basket mean)
"""
import numpy as np


def gamma2(record: dict) -> float:
    """Sampling overhead gamma^2 of one candidate."""
    if "gamma2" in record:                                # v3.1 records store it directly
        return float(record["gamma2"])
    return 2.0 ** float(record["log2_sampling_overhead"])  # otherwise rebuild it from log2


def bias_variance(circ: dict) -> np.ndarray:
    """Return an array of shape (n_candidates, 2): column 0 = b2, column 1 = v."""
    ideal = np.asarray(circ["obs_ideal"], float)          # mu*_S, one value per observable
    rows = []
    for r in circ["records"]:
        mu = np.asarray(r["obs_mu"], float)               # noisy expectation per observable
        b2 = float(np.mean((mu - ideal) ** 2))
        var = gamma2(r) - mu ** 2                         # per-observable per-shot variance
        v = float(np.mean(np.clip(var, 0.0, None)))       # negatives only from rounding
        rows.append((b2, v))
    return np.array(rows)


def mse(bv: np.ndarray, N: float) -> np.ndarray:
    """Expected squared error of every candidate at a budget of N shots."""
    return bv[:, 0] + bv[:, 1] / N


def oracle_set(bv: np.ndarray, N: float, tol: float = 1e-9) -> np.ndarray:
    """Indices of all candidates that are optimal at budget N (ties included)."""
    m = mse(bv, N)
    return np.flatnonzero(m <= m.min() * (1 + tol))
