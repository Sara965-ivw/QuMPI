"""No-cut option (decision A4): piecewise labelling of a separable circuit."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "datagen"):
    sys.path.insert(0, str(p))


def _separable():
    from quest_phase1.circuits import Circuit
    c = Circuit(6)
    for a, b in ((0, 1), (1, 2), (3, 4), (4, 5)):
        c.add("h", a); c.add("cx", (a, b)); c.add("rz", b, 0.3)
    return c


def test_pieces_and_tensor_order():
    from quest_phase1.nocut import _tensor, pieces, subcircuit
    from quest_phase1.simulators import probabilities
    c = _separable()
    parts = pieces(c)
    assert parts == [[0, 1, 2], [3, 4, 5]]
    p = _tensor([probabilities(subcircuit(c, qs)) for qs in parts], parts, c.n_qubits)
    assert np.allclose(p, probabilities(c), atol=1e-12)


def test_noiseless_label_is_exact_and_noisy_label_is_lower():
    from quest_phase1.labeling_v2 import make_device
    from quest_phase1.nocut import label_nocut
    ctx = make_device("fake_brisbane", seed=1234)
    c = _separable()
    ideal = label_nocut(ctx.ideal_copy(), c, max_width=3)
    assert ideal["oracle_tvd"] < 1e-12 and abs(ideal["pst_hellinger"] - 1) < 1e-12
    assert ideal["gamma2"] == 1.0 and ideal["log2_sampling_overhead"] == 0.0
    noisy = label_nocut(ctx, c, max_width=3)
    assert 0.5 < noisy["pst_hellinger"] < 1.0


def test_connected_circuit_is_rejected():
    from quest_phase1.circuits import Circuit
    from quest_phase1.labeling_v2 import make_device
    from quest_phase1.nocut import label_nocut
    c = Circuit(4)
    for a in range(3):
        c.add("cx", (a, a + 1))
    with pytest.raises(ValueError):
        label_nocut(make_device("fake_brisbane", seed=1234), c, max_width=3)
