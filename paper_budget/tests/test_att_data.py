"""Tests for paper_budget.att_data (PLAN_gt_att.md, step 2: tests T1 and T2)."""
import math

import numpy as np
import pytest

from paper_budget.att_data import attenuation, BIG
from paper_budget.config import DATA_V2


def test_attenuation_pure_damping():
    ideal = np.array([0.8, -0.5, 0.05])            # third observable is not informative
    f = np.array([0.9, 0.5])
    mu = f[:, None] * ideal[None, :]
    assert np.allclose(attenuation(mu, ideal), -np.log(f))


def test_attenuation_ignores_uninformative_and_can_be_negative():
    ideal = np.array([0.6, 0.1])
    mu = np.array([[0.63, -5.0]])                   # f = 1.05 on the informative one
    assert math.isclose(attenuation(mu, ideal)[0], -math.log(1.05))


def test_attenuation_nan_without_informative_observable():
    assert np.all(np.isnan(attenuation(np.ones((2, 2)) * 0.01, np.array([BIG, -BIG]))))


def _train_files(n=3):
    d = DATA_V2 / "train_v31"
    return sorted(p for p in d.glob("*.json") if "obs_ideal" in p.read_text())[:n] if d.exists() else []


@pytest.mark.skipif(not _train_files(), reason="train_v31 not available")
def test_T1_label_and_esp_equal_headroom_load():
    """The label and ESP are the ones the design analyses used (headroom.load)."""
    from paper_budget.analysis.headroom import load
    from paper_budget.att_data import load_circuit
    for p in _train_files():
        c, h = load_circuit(p, "train"), load(p)
        assert np.allclose(c.att, h["att"], equal_nan=True)
        assert np.allclose(c.e, h["e"])
        assert np.allclose(c.g2, h["g2"])
        assert np.allclose(np.column_stack([c.b2, c.v]), h["bv"])


@pytest.mark.skipif(len(_train_files(50)) < 10, reason="too few train_v31 files yet")
def test_T2_split_is_circuit_level_and_complete():
    from paper_budget.att_data import load_att_split, split_train_val
    C = [c for c in load_att_split("train_v31", "train") if c.usable_train()]
    tr, va = split_train_val(C)
    ids_tr, ids_va = {c.cid for c in tr}, {c.cid for c in va}
    assert not ids_tr & ids_va
    assert ids_tr | ids_va == {c.cid for c in C}
    tr2, va2 = split_train_val(C)                    # deterministic
    assert [c.cid for c in va2] == [c.cid for c in va]


def test_holdout3_stays_blinded(monkeypatch):
    from paper_budget.att_data import load_att_split
    monkeypatch.delenv("QUEST_UNBLIND", raising=False)
    with pytest.raises(PermissionError):
        load_att_split("holdout3")
