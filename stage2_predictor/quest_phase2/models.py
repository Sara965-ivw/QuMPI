"""Graph Transformer (Dwivedi and Bresson 2020) and a GIN baseline for cut ranking.

Both models read the same inputs: per-gate static features, per-candidate
features (side of each gate, gate-cut and wire-cut markers) and cut-level
features. They score every candidate of one circuit in a single batch.

Graph Transformer
  * Laplacian eigenvector positional encoding (k = 8), random sign flips in
    training because eigenvector signs are arbitrary (Dwivedi and Bresson);
  * global multi-head self-attention over all gates, with a learned bias per
    head for the bucketed shortest-path distance and for wire adjacency
    (Graphormer-style structural bias), so attention sees the circuit
    topology while still reaching gates any number of hops apart. This is
    the stated reason for preferring attention over local message passing:
    entanglement in a circuit is long-range;
  * pre-LayerNorm blocks, 3 layers, 4 heads, width 64 (proposal Section 6.1).
GIN (Xu et al. 2019): sum aggregation over wire neighbours only, the same
width, depth, readout and loss. It isolates what global attention adds.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_CAND, N_GLOBAL, N_GLOBAL2, N_HAND, N_STATIC, SPD_MAX


class _Readout(nn.Module):
    """mean+max pooling (run 1). With additive=True a per-gate contribution is
    also SUMMED over gates: noise accumulates over gates (log F is roughly a sum
    of per-gate log fidelities), so a size-extensive term is the physically
    appropriate inductive bias for extrapolating to wider circuits."""
    def __init__(self, d: int, dropout: float, n_global: int = N_GLOBAL, n_hand: int = 0,
                 additive: bool = False):
        super().__init__()
        self.additive = additive
        self.head = nn.Sequential(nn.Linear(2 * d + n_global + n_hand, d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d, 1))
        if additive:
            self.node = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h, g_feat, hand=None):          # h: (C, N, d)
        parts = [h.mean(1), h.max(1).values, g_feat]
        if hand is not None:
            parts.append(hand)
        out = self.head(torch.cat(parts, dim=-1)).squeeze(-1)
        if self.additive:
            out = out + 0.05 * self.node(h).squeeze(-1).sum(1)   # fixed scale, still extensive
        return out                                     # (C,)


class GraphTransformer(nn.Module):
    def __init__(self, d: int = 64, heads: int = 4, layers: int = 3, k_pe: int = 8,
                 dropout: float = 0.1, use_pe: bool = True, use_bias: bool = True,
                 pe: str = "lap", additive: bool = False, size_free: bool = False,
                 hybrid: bool = False, delta: bool = False, n_cand: int = N_CAND):
        super().__init__()
        self.heads, self.use_pe, self.use_bias = heads, use_pe, use_bias
        self.pe_kind, self.size_free, self.hybrid = pe, size_free, hybrid
        self.delta = delta
        self.inp = nn.Linear(N_STATIC + n_cand, d)      # n_cand > N_CAND only for Card B 'both'
        self.pe = nn.Linear(k_pe, d)
        self.spd_bias = nn.Embedding(SPD_MAX + 1, heads)
        self.adj_bias = nn.Linear(1, heads)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d),
                "attn": nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True),
                "ln2": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(2 * d, d)),
            }))
        self.drop = nn.Dropout(dropout)
        self.out = _Readout(d, dropout, N_GLOBAL2 if size_free else N_GLOBAL,
                            N_HAND if hybrid else 0, additive)
        if delta:
            # Delta-learning (Ramakrishnan et al. 2015): score = w * log ESP + f(graph).
            # The last layers of f start at zero, so an untrained model ranks exactly
            # as layout-ESP does; training only learns the correction to ESP.
            # w is learned because log ESP and the label live on different scales.
            self.esp_w = nn.Parameter(torch.tensor(1.0))
            nn.init.zeros_(self.out.head[-1].weight); nn.init.zeros_(self.out.head[-1].bias)
            if additive:
                nn.init.zeros_(self.out.node[-1].weight); nn.init.zeros_(self.out.node[-1].bias)

    def forward(self, batch: dict) -> torch.Tensor:
        x = torch.cat([batch["x_static"].unsqueeze(0).expand(batch["x_cand"].shape[0], -1, -1),
                       batch["x_cand"]], dim=-1)                       # (C, N, F)
        h = self.inp(x)
        if self.use_pe:
            if self.pe_kind == "rw":
                pe = batch["rw_pe"]
            else:
                pe = batch["lap_pe"]
                if self.training:                                      # sign invariance
                    pe = pe * (torch.randint(0, 2, (1, pe.shape[1]), device=pe.device) * 2 - 1)
            h = h + self.pe(pe).unsqueeze(0)
        C, N, _ = h.shape
        mask = None
        if self.use_bias:
            b = self.spd_bias(batch["spd"]) + self.adj_bias(batch["adj"].unsqueeze(-1))  # (N, N, H)
            mask = b.permute(2, 0, 1).unsqueeze(0).expand(C, -1, -1, -1).reshape(C * self.heads, N, N)
        for blk in self.blocks:
            z = blk["ln1"](h)
            a, _ = blk["attn"](z, z, z, attn_mask=mask, need_weights=False)
            h = h + self.drop(a)
            h = h + self.drop(blk["ff"](blk["ln2"](h)))
        f = self.out(h, batch["g_feat2"] if self.size_free else batch["g_feat"],
                     batch["hand_z"] if self.hybrid else None)
        if self.delta:
            return self.esp_w * batch["esp"] + f
        return f


class GIN(nn.Module):
    def __init__(self, d: int = 64, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(N_STATIC + N_CAND, d)
        self.eps = nn.Parameter(torch.zeros(layers))
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
                                   for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.out = _Readout(d, dropout)

    def forward(self, batch: dict) -> torch.Tensor:
        x = torch.cat([batch["x_static"].unsqueeze(0).expand(batch["x_cand"].shape[0], -1, -1),
                       batch["x_cand"]], dim=-1)
        h = self.inp(x)
        adj = batch["adj"]                                             # wire multiplicities
        for i, (mlp, ln) in enumerate(zip(self.mlps, self.norms)):
            agg = torch.einsum("ij,cjd->cid", adj, h)
            h = h + self.drop(mlp(ln((1 + self.eps[i]) * h + agg)))
        return self.out(h, batch["g_feat"])


def listnet_loss(scores: torch.Tensor, y_z: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """ListNet top-one cross-entropy between the label and score distributions
    of one circuit (Cao et al. 2007). Labels are within-circuit z-scores, so the
    objective is invariant to the between-circuit fidelity level that carries
    about 90 per cent of the raw label variance."""
    target = F.softmax(y_z / tau, dim=0)
    return -(target * F.log_softmax(scores, dim=0)).sum()


def gap_pairwise_loss(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gap-weighted pairwise logistic loss (RankNet, Burges et al. 2005, with
    LambdaRank-style pair weights, Burges 2010) on the RAW labels of one circuit.

    For every pair (i, j) with y_i > y_j the loss is log(1 + exp(-(s_i - s_j))),
    weighted by |y_i - y_j| and normalised by the circuit's total weight. This is
    the smooth counterpart of the primary metric (gap-weighted Kendall tau):
    pairs whose labels differ by almost nothing barely contribute, unlike
    ListNet on within-circuit z-scores, which stretches every circuit to unit
    spread and so trains on noise-level differences."""
    i, j = torch.triu_indices(len(y), len(y), 1, device=y.device)
    dy = y[i] - y[j]
    w = dy.abs()
    if float(w.sum()) <= 0:
        return scores.sum() * 0.0
    ds = torch.sign(dy) * (scores[i] - scores[j])
    return (w * F.softplus(-ds)).sum() / w.sum()


def to_batch(g, device) -> dict:
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    return {"x_static": t(g.x_static), "x_cand": t(g.x_cand), "g_feat": t(g.g_feat),
            "adj": t(g.adj), "spd": t(g.spd, torch.long), "lap_pe": t(g.lap_pe),
            "rw_pe": t(g.rw_pe), "g_feat2": t(g.g_feat2),
            "hand_z": t(g.hand_z if g.hand_z is not None else np.zeros((g.n_cand, N_HAND))),
            "y_z": t(g.y_z), "y": t(g.y),
            "esp": t(g.esp if g.esp is not None else np.zeros(g.n_cand))}


# Variants compared in the width-selection protocol (fixed before running it).
VARIANTS = {
    "gt_run1":   dict(),                                                    # run 1 as reported
    "gt_sum":    dict(additive=True),                                       # + size-extensive readout
    "gt_size":   dict(additive=True, size_free=True, pe="rw"),              # + no absolute size cues
    "gt_hybrid": dict(additive=True, size_free=True, pe="rw", hybrid=True),  # + per-side error sums
    "gt_delta":  dict(additive=True, size_free=True, pe="rw", delta=True),  # gt_size + ESP offset (Delta)
}
