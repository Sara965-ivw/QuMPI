"""Corpus -> graph tensors for RQ1 (one shared representation for every model).

Each labelled circuit becomes one CircuitGraph:
  * nodes   = gates, in circuit order (the Gate Dependency Graph);
  * edges   = wire dependencies stored as a MULTIGRAPH, one edge per wire
              (u, v, qubit). Two gates that follow each other on two wires keep
              two edges; this fixes the lost-edge defect of the Phase 0
              dual_graph.py, where an nx.DiGraph merged them;
  * static node features (gate type, angle, device calibration of the uncut
    layout, time position);
  * one candidate-feature block per labelled cut (side of each gate, gate-cut
    flag, wire-cut endpoints) plus cut-level features (g, w, widths, overhead);
  * labels: Hellinger fidelity per cut and its within-circuit z-score.

Circuits are rebuilt from (family, generator arguments, seed) stored in the
corpus, so the graph is exactly the circuit that was labelled.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATAGEN = Path(__file__).resolve().parents[1] / "datagen"
if str(DATAGEN) not in sys.path:
    sys.path.insert(0, str(DATAGEN))

from quest_phase1 import generators as G  # noqa: E402

GATE_VOCAB = ["h", "x", "y", "z", "s", "t", "id", "rx", "ry", "rz", "cx", "cz", "rzz"]
_GEN = {"clustered_random": G.clustered_random_circuit, "twolocal": G.twolocal_circuit,
        "qaoa_sbm": G.qaoa_sbm_circuit, "random": G.random_circuit,
        "qaoa_maxcut": G.qaoa_maxcut_circuit}
FAMILIES = ["clustered_random", "qaoa_sbm", "twolocal"]
SPD_MAX = 8                       # shortest-path distances bucketed 0..8 (8 = farther/unreachable)
N_STATIC = len(GATE_VOCAB) + 2 + 7 + 2
N_CAND_BASE = 5
N_LAYOUT = 4                      # fragment-layout calibration channels (hardware.py)
N_CAND = N_CAND_BASE + N_LAYOUT
N_GLOBAL = 6 + len(FAMILIES)
N_GLOBAL2 = 6 + len(FAMILIES)     # size-free cut features (no n, absolute widths)
N_HAND_BASE = 19
N_HAND = N_HAND_BASE + 4          # + layout-ESP of each fragment, mean T1 of each fragment


def rebuild_circuit(meta: dict):
    gen = _GEN[meta["family"]]
    params = inspect.signature(gen).parameters
    return gen(**{k: v for k, v in meta.items() if k in params})[0]


@dataclass
class CircuitGraph:
    cid: str
    family: str
    n_qubits: int
    split: str
    x_static: np.ndarray          # (N, N_STATIC)
    x_cand: np.ndarray            # (C, N, N_CAND)
    g_feat: np.ndarray            # (C, N_GLOBAL)
    adj: np.ndarray               # (N, N) wire-edge multiplicity, symmetric
    spd: np.ndarray               # (N, N) bucketed shortest-path distance
    lap_pe: np.ndarray            # (N, k) Laplacian eigenvectors
    y: np.ndarray                 # (C,) Hellinger fidelity
    y_z: np.ndarray               # (C,) within-circuit z-score (0 if constant)
    cut_g: np.ndarray             # (C,) gate cuts per candidate
    hand: np.ndarray = None       # (C, F_hand) features for gradient boosting
    meta: dict = field(default_factory=dict)
    records: list = field(default_factory=list)
    rw_pe: np.ndarray = None      # (N, k) random-walk return probabilities (size-transferable)
    g_feat2: np.ndarray = None    # (C, N_GLOBAL2) size-free cut features
    hand_z: np.ndarray = None     # (C, N_HAND) hand features standardised on the training split
    esp: np.ndarray = None        # (C,) layout-ESP of the cut (log), for the ESP baseline
    esp_uniform: np.ndarray = None  # (C,) same with device-average calibration (placement only)

    @property
    def n_cand(self) -> int:
        return len(self.y)


def _log_scaled(v, scale):
    return math.log10(max(v, 1e-12)) / scale


def _layers(circ) -> list:
    """ASAP layer index of every gate (time position feature)."""
    depth_q = [0] * circ.n_qubits
    out = []
    for g in circ.gates:
        lvl = max(depth_q[q] for q in g.qubits)
        out.append(lvl)
        for q in g.qubits:
            depth_q[q] = lvl + 1
    return out


def _spd(adj_bool: np.ndarray) -> np.ndarray:
    from scipy.sparse.csgraph import shortest_path
    d = shortest_path(adj_bool.astype(float), directed=False, unweighted=True)
    d[~np.isfinite(d)] = SPD_MAX
    return np.minimum(d, SPD_MAX).astype(np.int64)


def _lap_pe(adj: np.ndarray, k: int = 8) -> np.ndarray:
    n = adj.shape[0]
    deg = adj.sum(1)
    with np.errstate(divide="ignore"):
        dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    lap = np.eye(n) - dinv[:, None] * adj * dinv[None, :]
    _, vecs = np.linalg.eigh(lap)
    pe = vecs[:, 1:k + 1]                       # drop the trivial eigenvector
    if pe.shape[1] < k:
        pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
    return pe.astype(np.float32)


def _rw_pe(adj: np.ndarray, k: int = 8) -> np.ndarray:
    """Random-walk structural encoding (Dwivedi et al. 2022): the probability of
    returning to a node after 1..k steps. It depends only on the local
    neighbourhood, so its meaning does not change with circuit size, unlike
    Laplacian eigenvectors whose basis is global and size-dependent."""
    deg = adj.sum(1, keepdims=True)
    P = np.divide(adj, deg, out=np.zeros_like(adj, dtype=float), where=deg > 0)
    out, M = [], np.eye(adj.shape[0])
    for _ in range(k):
        M = M @ P
        out.append(np.diag(M))
    return np.stack(out, 1).astype(np.float32)


def build_graph(path: Path, split: str, k_pe: int = 8, aggregation: str = "both"):
    """aggregation: 'both' (default) / 'min' / 'mean' = calibration of the UNCUT
    layout on the gate nodes; 'none' = no hardware information; 'layout' =
    calibration of the qubits each fragment actually uses, per candidate
    (hardware.py), with the uncut-layout channels zeroed."""
    c = json.loads(Path(path).read_text())
    recs = c["records"]
    if not recs:
        return None
    meta = c["circuit_meta"]
    circ = rebuild_circuit(meta)
    n, N = circ.n_qubits, len(circ.gates)
    feats = c["node_features"]
    if len(feats) != N:
        raise ValueError(f"{path.name}: {len(feats)} feature rows for {N} gates")
    layers = _layers(circ)
    depth = max(layers) + 1

    xs = np.zeros((N, N_STATIC), np.float32)
    for i, (gt, f) in enumerate(zip(circ.gates, feats)):
        xs[i, GATE_VOCAB.index(gt.name)] = 1.0
        o = len(GATE_VOCAB)
        if gt.params:
            xs[i, o], xs[i, o + 1] = math.sin(gt.params[0]), math.cos(gt.params[0])
        o += 2
        hw = [_log_scaled(f["t1_min"], 3), _log_scaled(f["t1_mean"], 3),
              _log_scaled(f["t2_min"], 3), _log_scaled(f["t2_mean"], 3),
              -_log_scaled(f["gate_error"], 4), f.get("coupled_on_device", 1.0),
              -_log_scaled(f.get("readout_eps_mean", 1e-2), 3)]
        if aggregation == "min":
            hw[1], hw[3] = hw[0], hw[2]
        elif aggregation == "mean":
            hw[0], hw[2] = hw[1], hw[3]
        elif aggregation in ("none", "layout", "layout_uniform", "layout_min", "layout_mean", "layout_both"):   # no uncut-layout hardware information
            hw = [0.0] * 7
        xs[i, o:o + 7] = hw
        o += 7
        xs[i, o] = layers[i] / max(depth - 1, 1)
        xs[i, o + 1] = 1.0 if gt.n_qubits == 2 else 0.0

    # wire multigraph (fixes the Phase 0 lost-edge defect)
    edges = [(u, v, q) for u, v, q in circ.dag_edges()]
    adj = np.zeros((N, N), np.float32)
    for u, v, _ in edges:
        adj[u, v] += 1.0
        adj[v, u] += 1.0
    spd = _spd(adj > 0)
    pe = _lap_pe((adj > 0).astype(float), k_pe)
    rw = _rw_pe((adj > 0).astype(float), k_pe)

    C = len(recs)
    # Card B: layout_both stores minimum and mean as separate columns (7 layout channels)
    lay_agg = {"layout_min": "min", "layout_mean": "mean", "layout_both": "both"}.get(aggregation, "mixed")
    from .hardware import AGG_CHANNELS
    xc = np.zeros((C, N, N_CAND_BASE + AGG_CHANNELS[lay_agg]), np.float32)
    gf = np.zeros((C, N_GLOBAL), np.float32)
    gf2 = np.zeros((C, N_GLOBAL2), np.float32)
    y = np.array([r["pst_hellinger"] for r in recs], np.float64)
    gates_on = [[i for i, gt in enumerate(circ.gates) if q in gt.qubits] for q in range(n)]
    from quest_phase1.mixed_cuts import MixedCut, plan_cut
    from .hardware import layout_features
    esp = np.zeros(C, np.float32)
    esp_u = np.zeros(C, np.float32)
    hand_hw = np.zeros((C, 4), np.float32)
    for j, r in enumerate(recs):
        plan = plan_cut(circ, MixedCut(frozenset(r["home_a"]),
                                       tuple(tuple(x) for x in r["wire_cuts"]), n))
        if plan is not None and "phys_a" in r:
            lf = layout_features(circ, r, plan, agg=lay_agg)
            lu = layout_features(circ, r, plan, uniform=True)
            esp[j] = lf["log_esp_a"] + lf["log_esp_b"]
            esp_u[j] = lu["log_esp_a"] + lu["log_esp_b"]
            if aggregation == "layout_uniform":      # calibration control: placement only
                lf = lu
            if aggregation in ("layout", "layout_uniform", "layout_min", "layout_mean", "layout_both"):
                xc[j, :, N_CAND_BASE:] = lf["gate"]
                hand_hw[j] = [lf["log_esp_a"], lf["log_esp_b"],
                              math.log10(max(lf["t1_a"], 1e-7)), math.log10(max(lf["t1_b"], 1e-7))]
    for j, r in enumerate(recs):
        home_a = set(r["home_a"])
        wire = {q: t for q, t in r["wire_cuts"]}
        cut_gates = set(r["gate_cut_indices"])
        for i, gt in enumerate(circ.gates):
            sides = []
            for q in gt.qubits:
                if q in wire:
                    sides.append(0.0 if i <= wire[q] else 1.0)
                else:
                    sides.append(0.0 if q in home_a else 1.0)
            xc[j, i, 0] = float(np.mean(sides))
            xc[j, i, 1] = 1.0 if i in cut_gates else 0.0
            xc[j, i, 2] = 1.0 if any(q in wire for q in gt.qubits) else 0.0
        for q, t in wire.items():          # endpoints of the cut wire edge
            nxt = [i for i in gates_on[q] if i > t]
            xc[j, t, 3] = 1.0
            if nxt:
                xc[j, nxt[0], 4] = 1.0
        gf[j, :6] = [r["g"] / 2, r["w"], r["width_a"] / n, r["width_b"] / n,
                     r["log2_sampling_overhead"] / 10, n / 24]
        gf[j, 6 + FAMILIES.index(meta["family"])] = 1.0 if meta["family"] in FAMILIES else 0.0
        gf2[j, :6] = [r["g"] / 2, r["w"], r["width_a"] / 13, r["width_b"] / 13,
                      abs(r["width_a"] - r["width_b"]) / 13, r["log2_sampling_overhead"] / 10]
        gf2[j, 6 + FAMILIES.index(meta["family"])] = 1.0
    sd = y.std()
    y_z = (y - y.mean()) / sd if sd > 1e-12 else np.zeros_like(y)

    hand = _hand_features(circ, feats, recs, n)
    if aggregation in ("none", "layout", "layout_uniform", "layout_min", "layout_mean", "layout_both"):   # no uncut-layout calibration in the hand set
        hand[:, [7, 8, 9, 10, 12, 13, 14, 15, 17]] = 0.0   # calibration columns; structure kept
    hand = np.concatenate([hand, hand_hw], 1)
    g = CircuitGraph(meta.get("circuit_id", Path(path).stem), meta["family"], n, split,
                     xs, xc, gf, adj, spd, pe, y.astype(np.float32), y_z.astype(np.float32),
                     np.array([r["g"] for r in recs]), hand, meta, recs, rw, gf2)
    g.esp = esp
    g.esp_uniform = esp_u
    return g


def _hand_features(circ, feats, recs, n) -> np.ndarray:
    """Per-candidate engineered features for the gradient-boosting baseline:
    cut structure plus calibration statistics of each side and of the cut gates."""
    rows = []
    for r in recs:
        home_a = set(r["home_a"])
        wire = {q: t for q, t in r["wire_cuts"]}
        side_err = {0: [], 1: []}
        side_t1 = {0: [], 1: []}
        side_2q = {0: 0, 1: 0}
        cut_err = []
        for i, (gt, f) in enumerate(zip(circ.gates, feats)):
            s = []
            for q in gt.qubits:
                s.append(0 if (i <= wire[q] if q in wire else q in home_a) else 1)
            if i in r["gate_cut_indices"]:
                cut_err.append(f["gate_error"])
                continue
            sd = s[0]
            side_err[sd].append(f["gate_error"])
            side_t1[sd].append(f["t1_min"])
            side_2q[sd] += gt.n_qubits == 2
        row = [r["g"], r["w"], r["width_a"], r["width_b"], max(r["width_a"], r["width_b"]),
               r["log2_sampling_overhead"], n]
        for sd in (0, 1):
            e = side_err[sd] or [0.0]
            t = side_t1[sd] or [0.0]
            row += [float(np.sum(e)), float(np.max(e)), float(np.mean(t)), float(np.min(t)),
                    side_2q[sd]]
        row += [float(np.sum(cut_err)) if cut_err else 0.0, len(cut_err)]
        rows.append(row)
    return np.array(rows, np.float32)


def _blind_guard(corpus_dir) -> None:
    """holdout3 is sequestered by the pre-registration: no model-development
    code may read it until Part B (frozen model + commit) is recorded."""
    if Path(corpus_dir).name == "holdout3" and os.environ.get("QUEST_UNBLIND") != "holdout3":
        raise PermissionError("holdout3 is blinded (PREREG_holdout3.md). Set "
                              "QUEST_UNBLIND=holdout3 only for the single confirmatory run.")


def load_split(corpus_dir: Path, split: str, limit: int = None, aggregation: str = "both") -> list:
    _blind_guard(corpus_dir)
    files = sorted(Path(corpus_dir).glob("*.json"))
    files = [f for f in files if not f.name.endswith(".g1_empty.json")]
    if limit:
        files = files[:limit]
    graphs = []
    for f in files:
        g = build_graph(f, split, aggregation=aggregation)
        if g is not None:
            graphs.append(g)
    return graphs


def stratified_split(graphs: list, val_frac: float = 0.2, seed: int = 2026) -> tuple:
    """Circuit-level split stratified by (family, width): candidates of one
    circuit never appear in both train and validation (no leakage)."""
    rng = np.random.default_rng(seed)
    cells = {}
    for g in graphs:
        cells.setdefault((g.family, g.n_qubits), []).append(g)
    train, val = [], []
    for key in sorted(cells):
        grp = cells[key]
        idx = rng.permutation(len(grp))
        n_val = max(1, int(round(val_frac * len(grp)))) if len(grp) >= 3 else 0
        for k, i in enumerate(idx):
            (val if k < n_val else train).append(grp[i])
    for g in val:
        g.split = "val"
    for g in train:
        g.split = "train"
    return train, val
