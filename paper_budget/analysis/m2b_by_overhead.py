"""Where does M2b's gain over M1 come from? (EXPLORATORY, post hoc; holdout3)

On the 162-circuit evaluation set M2b had 2.4-2.6% lower regret than M1, but on the 121 circuits it
shares with the October comparison set the two were equal (RESULTS_m2b_holdout3.md, compare_tools_m2b).
Hypothesis (formed after seeing those numbers): the gain sits in circuits whose candidate cuts all have
the same sampling overhead, where the choice depends on the noise ranking alone. This script splits the
162 circuits by (in the October set or not) x (one overhead or several) and reports M2b/M1 per group.
Read-only: writes nothing.

  QUEST_UNBLIND=holdout3 python -m paper_budget.analysis.m2b_by_overhead
"""
import json
import os

import numpy as np
import torch

from paper_budget.att_data import load_circuit
from paper_budget.config import DATA_V2, RUNS
from paper_budget.eval_gt_att import log_regret
from paper_budget.eval_holdout4 import RUN, SEEDS, _pred, verify_frozen
from paper_budget.models_att import ResidualGT, clip_att, predict_m1


def main():
    assert os.environ.get("QUEST_UNBLIND") == "holdout3", "set QUEST_UNBLIND=holdout3"
    verify_frozen()
    cfg = json.loads((RUN / "main" / "config.json").read_text())
    ab, w, a_min = (cfg["alpha"], cfg["beta"]), np.asarray(cfg["w"]), cfg["a_min"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = []
    for s in SEEDS:
        n = ResidualGT().to(dev)
        n.load_state_dict(torch.load(RUN / "m2b" / f"M2b_seed{s}.pt", map_location=dev)["state_dict"])
        n.eval()
        nets.append(n)
    cmp_ids = {r["cid"] for r in json.load(open(RUNS / "compare_m2b" / "per_circuit.json"))}
    groups = {}
    for p in sorted((DATA_V2 / "holdout3").glob("*.json")):
        if p.name.endswith(".g1_empty.json"):
            continue
        c = load_circuit(p, "holdout3")
        if c is None or not c.usable_eval():
            continue
        base = predict_m1(c, ab, w)
        m1 = clip_att(base, a_min)[0]
        m2b = clip_att(np.mean([_pred(n, c, base, dev) for n in nets], axis=0), a_min)[0]
        d = log_regret(c, m2b) - log_regret(c, m1)
        n_k = len(set(np.round(np.log2(c.g2), 9)))
        key = ("in October set" if p.stem in cmp_ids else "not in October set",
               "one overhead" if n_k == 1 else "several overheads")
        groups.setdefault(key, []).append(d)
    allv = np.vstack([np.array(v) for v in groups.values()])
    print(f"all evaluation circuits: n {len(allv)}  M2b/M1 at 1e8, 1e9: "
          f"{np.exp(allv[:, 5].mean()):.4f} {np.exp(allv[:, 6].mean()):.4f}  (should match 0.9756 0.9740)")
    for k, v in sorted(groups.items()):
        v = np.array(v)
        print(f"{k[0]:<20} {k[1]:<18} n {len(v):3d}  M2b/M1 at 1e8, 1e9: "
              f"{np.exp(v[:, 5].mean()):.4f} {np.exp(v[:, 6].mean()):.4f}  "
              f"different choices {int((np.abs(v[:, 6]) > 1e-12).sum())}")


if __name__ == "__main__":
    main()
