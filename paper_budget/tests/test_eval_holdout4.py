"""holdout4 stays blinded in the confirmatory evaluation."""
import sys

import pytest


@pytest.mark.parametrize("split", ["holdout4", "holdout3"])
def test_refuses_without_unblinding(monkeypatch, split):
    from paper_budget import eval_holdout4
    monkeypatch.delenv("QUEST_UNBLIND", raising=False)
    monkeypatch.setattr(sys, "argv", ["eval_holdout4", "--test", split])
    with pytest.raises(AssertionError, match="QUEST_UNBLIND"):
        eval_holdout4.main()


def test_frozen_list_has_15_checkpoints():
    from paper_budget.eval_holdout4 import FROZEN
    assert len(FROZEN) == 15 and all(len(h) == 64 for h in FROZEN.values())
