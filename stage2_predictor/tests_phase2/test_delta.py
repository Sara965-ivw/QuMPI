"""Invariants of the Delta model, the gap-weighted pairwise loss and GBM-Delta.
They use synthetic inputs only, so they run without the corpus."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quest_phase2.data import N_CAND, N_GLOBAL, N_GLOBAL2, N_HAND, N_STATIC  # noqa: E402
from quest_phase2.models import VARIANTS, GraphTransformer, gap_pairwise_loss  # noqa: E402


def _batch(C=5, N=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    adj = (torch.rand(N, N, generator=g) > 0.6).float()
    adj = ((adj + adj.T) > 0).float()
    return {"x_static": torch.rand(N, N_STATIC, generator=g), "x_cand": torch.rand(C, N, N_CAND, generator=g),
            "g_feat": torch.rand(C, N_GLOBAL, generator=g), "g_feat2": torch.rand(C, N_GLOBAL2, generator=g),
            "adj": adj, "spd": torch.randint(0, 9, (N, N), generator=g), "lap_pe": torch.rand(N, 8, generator=g),
            "rw_pe": torch.rand(N, 8, generator=g), "hand_z": torch.zeros(C, N_HAND),
            "esp": -torch.rand(C, generator=g), "y": 0.9 + 0.1 * torch.rand(C, generator=g),
            "y_z": torch.zeros(C)}


def test_untrained_delta_equals_esp():
    """Zero-initialised correction: before training, the score IS log ESP."""
    m = GraphTransformer(**VARIANTS["gt_delta"]).eval()
    b = _batch()
    with torch.no_grad():
        assert torch.allclose(m(b), b["esp"], atol=1e-6)


def test_delta_correction_is_trainable():
    """The zero init must not block learning: one step changes the scores."""
    torch.manual_seed(0)
    m = GraphTransformer(**VARIANTS["gt_delta"])
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    b = _batch()
    gap_pairwise_loss(m(b), b["y"]).backward()
    opt.step()
    m.eval()
    with torch.no_grad():
        assert not torch.allclose(m(b), b["esp"], atol=1e-6)


def test_pairwise_loss_prefers_correct_order_and_ignores_ties():
    y = torch.tensor([0.99, 0.95, 0.90])
    good = gap_pairwise_loss(torch.tensor([3.0, 2.0, 1.0]), y)
    bad = gap_pairwise_loss(torch.tensor([1.0, 2.0, 3.0]), y)
    assert good < bad
    # all labels tied: nothing to learn, loss is exactly zero
    assert float(gap_pairwise_loss(torch.tensor([1.0, 2.0, 3.0]), torch.ones(3))) == 0.0


def test_pairwise_loss_weights_by_gap():
    """Misordering a pair with a large label gap costs more than a tiny gap."""
    s = torch.tensor([1.0, 0.0, -5.0])     # ranks 0 above 1 above 2
    # in both cases candidate 1 is truly better than 0 (one wrong pair); 2 is clearly worst
    big = gap_pairwise_loss(s, torch.tensor([0.90, 0.99, 0.70]))     # wrong pair, gap 0.09
    tiny = gap_pairwise_loss(s, torch.tensor([0.989, 0.99, 0.70]))   # wrong pair, gap 0.001
    assert big > tiny


def test_gbm_delta_falls_back_to_esp():
    """If log F - log ESP is the same for every candidate, GBM-Delta ranks as ESP."""
    from quest_phase2.baselines import GBMDeltaRanker
    rng = np.random.default_rng(0)
    graphs = []
    for _ in range(20):
        esp = -rng.random(6)
        graphs.append(SimpleNamespace(hand=rng.random((6, 4)), esp=esp, y=np.exp(esp + 0.3)))
    r = GBMDeltaRanker(0).fit(graphs)
    g = graphs[0]
    assert np.argsort(r.score(g)).tolist() == np.argsort(g.esp).tolist()
