"""Tests for the pure functions of paper_budget.eval_gt_att."""
from types import SimpleNamespace

import numpy as np

from paper_budget.eval_gt_att import boot_ratio, fit_metrics, gm, log_regret, wilcoxon_p


def _circuit(f, g2, ideal):
    f, g2, ideal = map(lambda z: np.asarray(z, float), (f, g2, ideal))
    mu = f[:, None] * ideal[None, :]
    return SimpleNamespace(att=-np.log(f), g2=g2, m=float(np.mean(ideal ** 2)),
                           b2=np.mean((mu - ideal) ** 2, axis=1),
                           v=np.mean(np.clip(g2[:, None] - mu ** 2, 0, None), axis=1), n_cand=len(f))


def test_true_attenuation_gives_zero_regret_under_pure_damping():
    c = _circuit([0.95, 0.80, 0.60], [256.0, 16.0, 9.0], [0.7, -0.4, 0.5])
    assert np.allclose(log_regret(c, c.att), 0.0)


def test_wrong_attenuation_gives_positive_regret_somewhere():
    c = _circuit([0.95, 0.80, 0.60], [256.0, 16.0, 9.0], [0.7, -0.4, 0.5])
    L = log_regret(c, c.att[::-1])                     # cleanest and noisiest cut swapped
    assert np.all(L >= 0) and np.any(L > 0)


def test_fit_metrics_perfect_prediction():
    C = [_circuit([0.9, 0.8, 0.7], [9, 16, 81], [0.5, 0.6]),
         _circuit([0.95, 0.6], [9, 9], [0.3, -0.8])]
    m = fit_metrics(C, [c.att for c in C])
    assert m["rmse"] < 1e-12 and abs(m["r2"] - 1) < 1e-12 and abs(m["r2_within"] - 1) < 1e-12
    assert abs(m["kendall_tau_b"] - 1) < 1e-12


def test_gm_bootstrap_and_wilcoxon_on_identical_inputs():
    L = np.log(np.full((20, 7), 1.1))
    assert np.allclose(gm(L), 1.1)
    assert np.allclose(boot_ratio(L, L), 1.0)
    assert wilcoxon_p(np.zeros(20)) == 1.0


def test_holdout3_guard_in_eval(monkeypatch):
    import pytest
    from paper_budget.eval_gt_att import load_test
    monkeypatch.delenv("QUEST_UNBLIND", raising=False)
    with pytest.raises(PermissionError):
        load_test("holdout3")
