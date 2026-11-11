"""Pure functions of paper_budget/analysis/compare_holdout4.py (no holdout data is read)."""
import sys

import numpy as np
import pytest

from paper_budget.analysis import compare_holdout4 as ch


def test_candidate_mse_is_bias_plus_variance_over_shots():
    M = ch.candidate_mse(np.array([0.01, 0.0]), np.array([1.0, 100.0]))
    assert M.shape == (2, len(ch.N))
    assert np.allclose(M[0], 0.01 + 1.0 / ch.N) and np.allclose(M[1], 100.0 / ch.N)


def test_budget_rule_with_true_attenuation_picks_the_best_candidate():
    """With the exact f and m, the rule's estimate is the true MSE, so it picks the minimum."""
    m, f, g2 = 0.5, np.array([0.99, 0.90, 0.80]), np.array([81.0, 9.0, 1.0])
    b2, v = (1 - f) ** 2 * m, g2 - f ** 2 * m
    M = ch.candidate_mse(b2, v)
    assert np.allclose(ch.budget_rule_mse(f, g2, m, M), M.min(0))


def test_paired_stats_ratio_ci_and_direction():
    rng = np.random.default_rng(0)
    L = np.log(1.10) + 0.02 * rng.standard_normal((60, len(ch.N)))
    t = ch.paired_stats(L)
    assert t["n"] == 60
    assert all(1.08 < r < 1.12 for r in t["geo_ratio"])
    assert all(lo < r < hi for lo, r, hi in zip(t["ci95"][0], t["geo_ratio"], t["ci95"][1]))
    assert all(p < 1e-6 for p in t["p_holm"]) and all(w == 60 for w in t["wins"])


def test_judge_reads_the_predictions_as_worded():
    def t(ratio, p):
        return {"geo_ratio": [ratio] * 7, "p_holm": [p] * 7}
    report = {"connected": {"M2b_vs_find_cuts": t(1.12, 0.01), "M2b_vs_qdislib": t(1.04, 0.2)},
              "all": {"M2b_vs_find_cuts": t(0.5, 0.001), "M2b_vs_qdislib": t(0.6, 0.001),
                      "M2b_vs_metis": t(1.3, 0.001), "M2b_vs_random": t(1.5, 0.001)}}
    v = ch.judge(report)
    assert v["P1"] is True and v["P2"] is False and v["P3"] is True and v["P4"] is True
    assert v["D1_first_budget_M2b_beats_find_cuts_connected"] == 1e3
    report["connected"]["M2b_vs_find_cuts"] = t(1.08, 0.01)          # below the 1.10 margin
    assert ch.judge(report)["P1"] is False
    report["all"]["M2b_vs_qdislib"]["geo_ratio"] = [1.1] + [0.6] * 6  # P3 is judged at 1e3 only
    assert ch.judge(report)["P3"] is False
    report["all"].pop("M2b_vs_qdislib")                                # Qdislib not run
    assert ch.judge(report)["P2"] is False and ch.judge(report)["P3"] is None


def test_refuses_without_unblinding(monkeypatch):
    monkeypatch.delenv("QUEST_UNBLIND", raising=False)
    monkeypatch.setattr(sys, "argv", ["compare_holdout4"])
    with pytest.raises(AssertionError, match="QUEST_UNBLIND"):
        ch.main()
