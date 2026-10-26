"""Card B: aggregation channels and the wider model input; the frozen model's input is unchanged."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_channels_and_frozen_input_width():
    from quest_phase2.data import N_CAND, N_CAND_BASE, N_STATIC
    from quest_phase2.hardware import AGG_CHANNELS, N_LAYOUT
    from quest_phase2.models import VARIANTS, GraphTransformer
    assert AGG_CHANNELS == {"mixed": 4, "min": 4, "mean": 4, "both": 7}
    assert N_CAND == N_CAND_BASE + N_LAYOUT == N_CAND_BASE + AGG_CHANNELS["mixed"]
    frozen = GraphTransformer(**VARIANTS["gt_size"])
    assert frozen.inp.in_features == N_STATIC + N_CAND          # frozen weights still load


def test_both_columns_forward():
    from quest_phase2.data import N_CAND_BASE, N_GLOBAL, N_GLOBAL2, N_HAND, N_STATIC
    from quest_phase2.models import VARIANTS, GraphTransformer
    n_cand = N_CAND_BASE + 7
    m = GraphTransformer(n_cand=n_cand, **VARIANTS["gt_size"]).eval()
    C, N = 4, 6
    b = {"x_static": torch.rand(N, N_STATIC), "x_cand": torch.rand(C, N, n_cand),
         "g_feat": torch.rand(C, N_GLOBAL), "g_feat2": torch.rand(C, N_GLOBAL2),
         "adj": torch.ones(N, N), "spd": torch.ones(N, N, dtype=torch.long),
         "lap_pe": torch.rand(N, 8), "rw_pe": torch.rand(N, 8), "hand_z": torch.zeros(C, N_HAND),
         "esp": torch.zeros(C), "y": torch.rand(C), "y_z": torch.zeros(C)}
    with torch.no_grad():
        assert m(b).shape == (C,)
