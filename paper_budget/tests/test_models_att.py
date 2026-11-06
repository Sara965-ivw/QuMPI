"""Tests for paper_budget.models_att (PLAN_gt_att.md, step 3: T3, T4, T5 and the closed-form fits)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from paper_budget.config import DATA_V2
from paper_budget.models_att import (A_MAX, DirectGT, ResidualGT, clip_att, fit_linear, fit_prior,
                                     predict_m0, predict_m1, predict_nn, prior)


def _fake(e, x, att):
    return SimpleNamespace(e=np.asarray(e, float), x=np.asarray(x, float), att=np.asarray(att, float))


def test_prior_recovers_exact_line():
    C = [_fake([0.5, 1.0, 2.0], np.zeros((3, 7)), [0.06, 0.11, 0.21]),
         _fake([1.5, 3.0], np.zeros((2, 7)), [0.16, 0.31])]
    a, b = fit_prior(C)
    assert abs(a - 0.01) < 1e-9 and abs(b - 0.10) < 1e-9


def test_linear_correction_is_within_circuit_and_recovers_slope():
    rng = np.random.default_rng(0)
    C = []
    for _ in range(30):
        e = rng.uniform(0.5, 3, 6)
        x = rng.normal(size=(6, 7))
        att = 0.5 * rng.normal() + 0.10 * e + 0.03 * (x[:, 3] - x[:, 3].mean())   # circuit level + slope
        C.append(_fake(e, x, att))
    ab = fit_prior(C)
    w = fit_linear(C, ab)
    for c in C:
        d = predict_m1(c, ab, w) - predict_m0(c, ab)
        assert abs(d.sum()) < 1e-9                       # correction does not move the circuit level
    dm = np.concatenate([(predict_m1(c, ab, w) - c.att) - (predict_m1(c, ab, w) - c.att).mean() for c in C])
    assert np.max(np.abs(dm)) < 1e-9                     # within-circuit part is fitted exactly


def test_T5_clip_bounds():
    a, n = clip_att(np.array([-1.0, 0.02, 10.0]), a_min=-0.05)
    assert np.allclose(a, [-0.05, 0.02, A_MAX]) and n == 2


def _real_circuit():
    d = DATA_V2 / "train_v31"
    if not d.exists():
        return None
    from paper_budget.att_data import load_circuit
    for p in sorted(d.glob("*.json")):
        c = load_circuit(p, "train")
        if c is not None and c.usable_train():
            return c
    return None


@pytest.mark.skipif(_real_circuit() is None, reason="no usable train_v31 circuit yet")
def test_T3_untrained_residual_model_equals_esp_prior_exactly():
    torch.manual_seed(0)
    c = _real_circuit()
    ab = (0.02, 0.1)
    out = predict_nn(ResidualGT(), c, ab)
    assert np.max(np.abs(out - prior(c, ab))) < 1e-6    # only float32 rounding of the prior


@pytest.mark.skipif(_real_circuit() is None, reason="no usable train_v31 circuit yet")
def test_T4_residual_sums_to_zero_after_perturbation():
    torch.manual_seed(1)
    c = _real_circuit()
    m = ResidualGT()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))           # break the zero initialisation
    ab = (0.02, 0.1)
    r = predict_nn(m, c, ab) - prior(c, ab)
    assert np.max(np.abs(r)) > 1e-6                      # the network now contributes
    assert abs(r.sum()) < 1e-4                           # but never changes the circuit mean


@pytest.mark.skipif(_real_circuit() is None, reason="no usable train_v31 circuit yet")
def test_direct_model_runs_and_has_gradients():
    torch.manual_seed(2)
    c = _real_circuit()
    m = DirectGT()
    from quest_phase2.models import to_batch
    out = m(to_batch(c.g, "cpu"))
    loss = torch.mean((out - torch.as_tensor(c.att, dtype=torch.float32)) ** 2)
    loss.backward()
    assert out.shape == (c.n_cand,)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.parameters())
