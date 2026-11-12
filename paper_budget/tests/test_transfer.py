"""Device-transfer test code (notes/PREREG_transfer_sherbrooke.md). Reads training data only.

The slow test relabels the smallest training circuit on fake_brisbane and checks that the
relabeller reproduces the frozen labeller's stored labels exactly.
"""
import json
import os
import sys

import numpy as np
import pytest

from paper_budget.config import DATA_V2
from paper_budget.datagen import relabel_device as rd
from paper_budget.device import current_device, use_device


def _smallest_train_file():
    files = [p for p in (DATA_V2 / "train_v31").glob("*.json") if not p.name.endswith(".g1_empty.json")]
    best = None
    for p in files:
        d = json.loads(p.read_text())
        if d.get("records") and "obs_ideal" in d:
            key = (d["circuit_meta"]["n_qubits"], len(d["records"]))
            if best is None or key < best[0]:
                best = (key, p)
    return best[1]


def test_default_device_tables_are_unchanged():
    import quest_phase1.labeling_v2 as lv2
    from quest_phase2 import hardware as hw
    original = lv2.make_device
    hw._device.cache_clear()
    native = hw._device()                         # the frozen function, untouched
    use_device("fake_brisbane")
    pinned = hw._device()
    assert native[1] == pinned[1] and native[2] == pinned[2]          # 1q and 2q error tables
    assert native[3] == pinned[3] and native[5] == pinned[5]          # mean 2q error, medians
    assert lv2.make_device is original            # restored
    assert current_device() == "fake_brisbane"


def test_switch_to_sherbrooke_and_back():
    from quest_phase2 import hardware as hw
    use_device("fake_sherbrooke")
    assert current_device() == "fake_sherbrooke"
    sher = hw._device()
    use_device("fake_brisbane")
    bris = hw._device()
    assert current_device() == "fake_brisbane"
    assert sher[1] != bris[1] or sher[2] != bris[2]   # different calibration tables


def test_sherbrooke_differs_only_in_calibration():
    """PREREG_transfer_sherbrooke.md section 2: same topology and two-qubit gate as fake_brisbane."""
    from quest_phase1.labeling_v2 import make_device
    b, s = (make_device(n, seed=1234).backend for n in ("fake_brisbane", "fake_sherbrooke"))
    undirected = lambda be: {tuple(sorted(e)) for e in be.coupling_map.get_edges()}
    assert b.num_qubits == s.num_qubits == 127
    assert undirected(b) == undirected(s)
    assert "ecr" in b.target.operation_names and "ecr" in s.target.operation_names


def test_unknown_device_is_refused():
    with pytest.raises(ValueError):
        use_device("fake_nowhere")


def test_output_split_name():
    assert rd.out_split("fake_sherbrooke") == "holdout4_sherbrooke"
    assert rd.out_split("fake_torino") == "holdout4_torino"


def test_rebuilt_candidates_match_their_records():
    """Every stored candidate of a training circuit is rebuilt with identical structure."""
    from quest_phase2.data import rebuild_circuit
    d = json.loads(_smallest_train_file().read_text())
    circ = rebuild_circuit(d["circuit_meta"])
    for r in d["records"]:
        rd.rebuild_plan(circ, r)                  # raises on any difference


def test_relabel_refuses_without_unblinding(monkeypatch):
    monkeypatch.delenv("QUEST_UNBLIND", raising=False)
    monkeypatch.setattr(sys, "argv", ["relabel_device", "--estimate"])
    with pytest.raises(AssertionError, match="QUEST_UNBLIND"):
        rd.main()


@pytest.mark.skipif(os.environ.get("PYTHONHASHSEED") != "0", reason="needs PYTHONHASHSEED=0 (layout ties)")
def test_relabel_on_the_same_device_reproduces_stored_labels():
    src = _smallest_train_file()
    d = json.loads(src.read_text())
    out = rd.relabel_circuit(src, "fake_brisbane")
    assert out["n_dropped_vs_source"] == 0
    assert len(out["records"]) == len(d["records"])
    for a, b in zip(d["records"], out["records"]):
        assert b["device"] == "fake_brisbane"
        assert a["phys_a"] == b["phys_a"] and a["phys_b"] == b["phys_b"]
        assert np.allclose(a["obs_mu"], b["obs_mu"], atol=1e-9, rtol=0)
        assert abs(a["pst_hellinger"] - b["pst_hellinger"]) < 1e-9
    assert out["obs_ideal"] == d["obs_ideal"]
