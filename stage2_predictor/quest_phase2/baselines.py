"""Non-learned and classical-ML baselines for RQ1.

random        : uniform scores, the floor (H1 falsification baseline).
min_overhead  : the objective of qiskit-addon-cutting's find_cuts: minimise the
                QPD sampling overhead; ties broken toward balanced fragments.
metis         : structural incumbent: a two-way METIS partition of the Qubit
                Interaction Graph (edge weight = two-qubit gate count); a cut
                scores by its partition's agreement with METIS (Jaccard).
                Falls back to Kernighan-Lin when pymetis is unavailable.
esp           : layout estimated success probability of the two fragments
                (hardware.py; approximates Nishio et al. 2020 on the fragment
                layouts): the standard hardware-aware heuristic, no training.
gbm           : gradient-boosted trees on engineered per-cut features (cut
                structure + calibration statistics per side), trained on the
                within-circuit z-score. Tests whether a deep model is needed.
"""
from __future__ import annotations

import numpy as np


def score_random(g, seed: int) -> np.ndarray:
    rng = np.random.default_rng(abs(hash((g.cid, seed))) % (2 ** 32))
    return rng.random(g.n_cand)


def score_min_overhead(g, seed: int = 0) -> np.ndarray:
    ov = np.array([r["log2_sampling_overhead"] for r in g.records])
    bal = np.array([max(r["width_a"], r["width_b"]) for r in g.records]) / g.n_qubits
    return -ov - 1e-3 * bal


def score_esp(g, seed: int = 0) -> np.ndarray:
    return np.asarray(g.esp, float)


def score_esp_uniform(g, seed: int = 0) -> np.ndarray:
    """Calibration control for ESP: placement and SWAP distances only."""
    return np.asarray(g.esp_uniform, float)


def _qig_partition(g) -> tuple:
    from .data import rebuild_circuit
    circ = rebuild_circuit(g.meta)
    n = circ.n_qubits
    w = {}
    for gt in circ.gates:
        if gt.n_qubits == 2:
            a, b = sorted(gt.qubits)
            w[(a, b)] = w.get((a, b), 0) + 1
    try:
        import pymetis
        adjacency = [[] for _ in range(n)]
        eweights = [[] for _ in range(n)]
        for (a, b), c in w.items():
            adjacency[a].append(b); eweights[a].append(c)
            adjacency[b].append(a); eweights[b].append(c)
        xadj, adjncy, ew = [0], [], []
        for q in range(n):
            adjncy += adjacency[q]; ew += eweights[q]; xadj.append(len(adjncy))
        _, parts = pymetis.part_graph(2, xadj=xadj, adjncy=adjncy, eweights=ew)
        side = {q for q in range(n) if parts[q] == 0}
        method = "metis"
    except Exception:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for (a, b), c in w.items():
            G.add_edge(a, b, weight=c)
        side, _ = nx.algorithms.community.kernighan_lin_bisection(G, weight="weight", seed=0)
        method = "kernighan_lin"
    return set(side), method


def score_metis(g, seed: int = 0) -> np.ndarray:
    part, _ = _qig_partition(g)
    other = set(range(g.n_qubits)) - part
    out = []
    for r in g.records:
        a = set(r["home_a"])
        j = max(len(a & part) / len(a | part), len(a & other) / len(a | other))
        out.append(j)
    return np.array(out)


def metis_method_used(graphs) -> str:
    return _qig_partition(graphs[0])[1] if graphs else "n/a"


class GBMRanker:
    def __init__(self, seed: int = 0):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                   max_leaf_nodes=31, l2_regularization=1e-3,
                                                   random_state=seed)

    def fit(self, graphs):
        X = np.concatenate([g.hand for g in graphs])
        y = np.concatenate([g.y_z for g in graphs])
        self.model.fit(X, y)
        return self

    def score(self, g) -> np.ndarray:
        return self.model.predict(g.hand)


class GBMDeltaRanker(GBMRanker):
    """Delta-learning with trees, the fair counterpart of gt_delta.

    Step 1, scale: log F and log ESP need not be on the same scale (Hellinger
    fidelity of a distribution decays more slowly than a success probability), so
    a single factor c is fitted by least squares on within-circuit centred values:
        c = sum(e~ * l~) / sum(e~ ^ 2),  l = log F, e = log ESP, ~ = centred per circuit.
    This mirrors the learned esp_w of gt_delta.
    Step 2, correction: trees learn the remainder r = l~ - c * e~.
    Score = c * log ESP + predicted r. With r predicted as 0 (and c > 0) this
    ranks exactly as ESP. If GBM-Delta matches gt_delta, the gain comes from the
    physics prior and not from attention; that comparison is the point of this class."""

    @staticmethod
    def _centred(g):
        l = np.log(np.clip(np.asarray(g.y, float), 1e-12, None))
        e = np.asarray(g.esp, float)
        return l - l.mean(), e - e.mean()

    def fit(self, graphs):
        le = [self._centred(g) for g in graphs]
        num = sum(float((l * e).sum()) for l, e in le)
        den = sum(float((e * e).sum()) for _, e in le)
        self.c = num / den if den > 0 else 1.0
        X = np.concatenate([g.hand for g in graphs])
        y = np.concatenate([l - self.c * e for l, e in le])
        self.model.fit(X, y)
        return self

    def score(self, g) -> np.ndarray:
        return self.c * np.asarray(g.esp, float) + self.model.predict(g.hand)
