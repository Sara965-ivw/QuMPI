"""Train the final M2b models (M1 + Transformer residual) for the holdout4 pre-registration.

M2b was selected by the pre-registered stage-1 diagnostics (notes/RESULTS_diag_gt.md). It is trained on
exactly the same train_v31 split and the same closed-form M1 (alpha, beta, w) as the main run
(runs/paper_budget/gt_att/main/config.json), so M0, M1, M2, M2b and M3 share one frozen baseline.
Training loop: diag_gt_att.train_nn (frozen, commit 25f8b5d), 5 seeds, run_rq1.HP.
Output: runs/paper_budget/gt_att/m2b/ (new folder; existing checkpoints are never overwritten),
with a SHA-256 of every checkpoint for the pre-registration.

  python -m paper_budget.train_m2b
"""
import hashlib
import json
import os
import time

import numpy as np
import torch

from paper_budget.att_data import load_att_split
from paper_budget.config import RUNS
from paper_budget.diag_gt_att import train_nn
from paper_budget.models_att import predict_m1
from paper_budget.train_gt_att import git_state
from run_rq1 import HP  # noqa: E402  (frozen)

MAIN = RUNS / "gt_att" / "main"
OUT = RUNS / "gt_att" / "m2b"
SEEDS = (0, 1, 2, 3, 4)


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    assert not os.environ.get("QUEST_UNBLIND"), "training never reads a holdout: unset QUEST_UNBLIND"
    assert os.environ.get("PYTHONHASHSEED") == "0", "export PYTHONHASHSEED=0 first"
    cfg = json.loads((MAIN / "config.json").read_text())
    ab, w = (cfg["alpha"], cfg["beta"]), np.asarray(cfg["w"])
    by_id = {c.cid: c for c in load_att_split(cfg["train_corpus"], "train")}
    tr, va = [by_id[i] for i in cfg["train_cids"]], [by_id[i] for i in cfg["val_cids"]]
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    git = git_state()
    print(f"M2b: train {len(tr)}, validation {len(va)} (split of {MAIN.name}); git {git['commit'][:8]}"
          f"{' (DIRTY)' if git['dirty'] else ''}; device {device}", flush=True)
    base = lambda c: predict_m1(c, ab, w)
    m1_val = float(np.mean([np.mean((base(c) - c.att) ** 2) for c in va]))
    print(f"validation loss of M1 (the starting point): {m1_val:.4e}", flush=True)
    t0, info = time.time(), {}
    for s in SEEDS:
        ck = OUT / f"M2b_seed{s}.pt"
        if ck.exists():
            print(f"  {ck.name} exists: not overwritten")
            info[ck.name] = sha256(ck)
            continue
        net = train_nn("M2b", s, tr, va, base, HP, device)
        with torch.no_grad():
            from quest_phase2.models import to_batch
            vl = float(np.mean([np.mean((net(to_batch(c.g, device),
                                             torch.as_tensor(base(c), dtype=torch.float32, device=device))
                                         .cpu().numpy() - c.att) ** 2) for c in va]))
        torch.save({"state_dict": net.state_dict(), "model": "M2b", "seed": s, "val_loss": vl,
                    "alpha": ab[0], "beta": ab[1], "w": w.tolist(), "git": git}, ck)
        info[ck.name] = sha256(ck)
        print(f"  M2b seed {s}: validation loss {m1_val:.4e} -> {vl:.4e}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    (OUT / "config.json").write_text(json.dumps({"from_run": str(MAIN), "train_cids": cfg["train_cids"],
                                                 "val_cids": cfg["val_cids"], "alpha": ab[0], "beta": ab[1],
                                                 "w": w.tolist(), "a_min": cfg["a_min"], "hp": HP,
                                                 "seeds": SEEDS, "git": git, "sha256": info,
                                                 "torch": torch.__version__}, indent=1))
    print("\nSHA-256 of the frozen models (copied into PREREG_holdout4.md):")
    for ck in sorted(MAIN.glob("*.pt")):
        print(f"  main/{ck.name}  {sha256(ck)}")
    for k, v in info.items():
        print(f"  m2b/{k}  {v}")


if __name__ == "__main__":
    main()
