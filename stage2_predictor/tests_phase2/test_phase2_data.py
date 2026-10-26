"""Phase 2 data invariants (run before every RQ1 training run)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quest_phase2.data import DATAGEN, _lap_pe, _spd  # noqa: E402,F401
from quest_phase2.metrics import circuit_metrics  # noqa: E402
from quest_phase1.circuits import Circuit  # noqa: E402


def test_double_wire_dependency_kept():
    """cx(0,1); cx(0,1) share TWO wires: the multigraph must keep both edges
    (the Phase 0 nx.DiGraph merged them and lost a cuttable wire)."""
    c = Circuit(2)
    c.add("cx", (0, 1))
    c.add("cx", (0, 1))
    edges = [(u, v) for u, v, _ in c.dag_edges()]
    assert edges.count((0, 1)) == 2


def test_spd_and_pe_shapes():
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], bool)
    d = _spd(adj)
    assert d[0, 2] == 2 and d[0, 0] == 0
    assert _lap_pe(adj.astype(float), 8).shape == (3, 8)


def test_metrics_perfect_and_constant():
    y = np.array([0.9, 0.95, 0.97])
    m = circuit_metrics(y, y)
    assert m["kendall"] == 1.0 and m["top1_regret"] == 0.0 and m["top1_hit"] == 1.0
    m = circuit_metrics(y, np.zeros(3))
    assert m["kendall"] == 0.0          # constant scores count as no ranking, not dropped


def test_holdout3_is_blinded(tmp_path, monkeypatch):
    """Pre-registration: holdout3 cannot be loaded during model development."""
    import pytest
    from quest_phase2.data import load_split
    monkeypatch.delenv("QUEST_UNBLIND", raising=False)
    (tmp_path / "holdout3").mkdir()
    with pytest.raises(PermissionError):
        load_split(tmp_path / "holdout3", "holdout3")
    monkeypatch.setenv("QUEST_UNBLIND", "holdout3")
    assert load_split(tmp_path / "holdout3", "holdout3") == []
