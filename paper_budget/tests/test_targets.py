"""targets.py must reproduce the thesis definition (budget_h3.py) exactly."""
import json
import os
import sys

import numpy as np
import pytest

from paper_budget.config import ROOT, DATA_V2
from paper_budget.targets import bias_variance, oracle_set

sys.path.insert(0, str(ROOT / "stage2_predictor" / "analysis"))
import budget_h3  # noqa: E402  (frozen thesis script, imported read-only)

H3 = sorted(p for p in (DATA_V2 / "holdout3").glob("*.json") if "g1_empty" not in p.name)


def test_synthetic_crossover():
    # A: large bias, small variance.  B: small bias, large variance (two CZ cuts).
    bv = np.array([[1e-2, 1.0],
                   [1e-4, 81.0]])
    assert oracle_set(bv, 1e3).tolist() == [0]    # few shots: variance dominates -> A
    assert oracle_set(bv, 1e6).tolist() == [1]    # many shots: bias dominates -> B


@pytest.mark.parametrize("path", H3, ids=lambda p: p.stem)
def test_matches_budget_h3(path):
    if os.environ.get("QUEST_UNBLIND") != "holdout3":
        pytest.skip("reading holdout3 needs QUEST_UNBLIND=holdout3")
    d = json.loads(path.read_text())
    if not d.get("records"):
        pytest.skip("audit file without labelled candidates")
    ours = bias_variance(d)
    b2, v, _ = budget_h3.terms(budget_h3.load_circuit(path))
    np.testing.assert_allclose(ours[:, 0], b2, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(ours[:, 1], v, rtol=1e-12, atol=1e-15)
