"""Attenuation models M0-M3 (notes/PLAN_gt_att.md, sections 1.3-1.5).

M0  ESP only                 a_hat = p = alpha + beta * e                     e = -ln ESP
M1  ESP + linear correction  a_hat = p + w . (z - mean_circuit z)             z = [e, 7 cut features]
M2  ESP + Transformer        a_hat = p + r(G) - mean_circuit r(G)             r zero-initialised
M3  direct Transformer       a_hat = g(G)
The prior (alpha, beta) and w are fitted on training circuits only. The Transformer is the frozen
RQ1 architecture (quest_phase2.models.GraphTransformer, variant gt_size, hyperparameters run_rq1.HP);
nothing in stage2_predictor/ is modified. Bounds are applied at prediction time only (clip_att).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import paper_budget.att_data  # noqa: F401  (puts stage2_predictor on sys.path)
from quest_phase2.models import VARIANTS, GraphTransformer, to_batch  # noqa: E402  (frozen)
from run_rq1 import HP  # noqa: E402  (frozen hyperparameters)

ARCH = "gt_size"
A_MAX = float(-np.log(0.01))            # f_hat >= 0.01


# ---------------------------------------------------------------- M0 and M1 (closed form)
def fit_prior(circuits) -> tuple[float, float]:
    """OLS of the label on e over all training candidates: a ~ alpha + beta * e."""
    e = np.concatenate([c.e for c in circuits])
    a = np.concatenate([c.att for c in circuits])
    beta, alpha = np.polyfit(e, a, 1)
    return float(alpha), float(beta)


def prior(c, ab) -> np.ndarray:
    return ab[0] + ab[1] * c.e


def _z_demeaned(c) -> np.ndarray:
    z = np.column_stack([c.e, c.x])
    return z - z.mean(axis=0)


def fit_linear(circuits, ab) -> np.ndarray:
    """w minimising sum_c sum_k (w.z~_k - u~_k)^2 with u = a - prior, ~ = within-circuit demeaned."""
    Z = np.vstack([_z_demeaned(c) for c in circuits])
    u = np.concatenate([(c.att - prior(c, ab)) - (c.att - prior(c, ab)).mean() for c in circuits])
    return np.linalg.lstsq(Z, u, rcond=None)[0]


def predict_m0(c, ab) -> np.ndarray:
    return prior(c, ab)


def predict_m1(c, ab, w) -> np.ndarray:
    return prior(c, ab) + _z_demeaned(c) @ w


# ---------------------------------------------------------------- M2 and M3 (Transformer)
def _backbone() -> GraphTransformer:
    return GraphTransformer(HP["d"], HP["heads"], HP["layers"], dropout=HP["dropout"], **VARIANTS[ARCH])


class ResidualGT(nn.Module):
    """M2: a_hat = prior + r(G) - mean over the circuit's candidates of r(G)."""

    def __init__(self):
        super().__init__()
        self.net = _backbone()
        out = self.net.out
        nn.init.zeros_(out.head[-1].weight)
        nn.init.zeros_(out.head[-1].bias)
        if getattr(out, "additive", False):
            nn.init.zeros_(out.node[-1].weight)
            nn.init.zeros_(out.node[-1].bias)

    def forward(self, batch: dict, prior_t: torch.Tensor) -> torch.Tensor:
        r = self.net(batch)
        return prior_t + (r - r.mean())


class DirectGT(nn.Module):
    """M3: a_hat = g(G), no prior, standard initialisation."""

    def __init__(self):
        super().__init__()
        self.net = _backbone()

    def forward(self, batch: dict, prior_t: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(batch)


def circuit_loss(a_hat: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Mean squared error over one circuit's candidates (each circuit weighs the same)."""
    return torch.mean((a_hat - a) ** 2)


@torch.no_grad()
def predict_nn(model: nn.Module, c, ab, device="cpu") -> np.ndarray:
    model.eval()
    b = to_batch(c.g, device)
    p = torch.as_tensor(prior(c, ab), dtype=torch.float32, device=device)
    return model(b, p).cpu().numpy().astype(float)


# ---------------------------------------------------------------- bounds (prediction time only)
def clip_att(a_hat: np.ndarray, a_min: float, a_max: float = A_MAX) -> tuple[np.ndarray, int]:
    """Clip to [a_min, a_max]; a_min = smallest training label. Returns (clipped, number clipped)."""
    out = np.clip(a_hat, a_min, a_max)
    return out, int(np.sum(out != a_hat))
