"""Train the attenuation models on the training split only (notes/PLAN_gt_att.md, step 4).

  python -m paper_budget.train_gt_att --smoke            # 2 epochs, 1 seed, separate folder
  python -m paper_budget.train_gt_att                    # full run (only after PREREG_gt_att.md is committed)

What it does
  1. loads train_v31 (att_data), keeps circuits usable for training, circuit-level split 80/20
     stratified by (family, width), seed 2026;
  2. fits M0 (alpha, beta) and M1 (w) on the training part, records a_min = smallest training label;
  3. trains M2 (residual Transformer) and M3 (direct Transformer), 5 seeds each, with the frozen RQ1
     hyperparameters (run_rq1.HP); loss = per-circuit MSE of the attenuation; early stopping on the
     validation loss. M2 starts from its initial state (= M0), so it can only be kept if training
     improves the validation loss.
Safety: refuses to run if QUEST_UNBLIND is set or a holdout is passed; never overwrites a checkpoint;
logs git commit, package versions and the full configuration next to the checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from paper_budget.att_data import load_att_split, split_train_val
from paper_budget.config import ROOT, RUNS, SPLIT_SEED
from paper_budget.models_att import (ARCH, DirectGT, ResidualGT, circuit_loss, fit_linear, fit_prior,
                                     predict_m0, predict_m1, prior)
from quest_phase2.models import to_batch  # noqa: E402  (frozen)
from run_rq1 import HP, set_seed  # noqa: E402  (frozen)

MODELS = {"M2": ResidualGT, "M3": DirectGT}


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def val_loss(model, circuits, ab, device) -> float:
    model.eval()
    with torch.no_grad():
        ls = [float(circuit_loss(model(to_batch(c.g, device),
                                       torch.as_tensor(prior(c, ab), dtype=torch.float32, device=device)),
                                 torch.as_tensor(c.att, dtype=torch.float32, device=device)))
              for c in circuits]
    return float(np.mean(ls))


def closed_form_loss(pred_fn, circuits) -> float:
    return float(np.mean([np.mean((pred_fn(c) - c.att) ** 2) for c in circuits]))


def train_one(name, seed, train, val, ab, hp, device, log):
    set_seed(seed)
    model = MODELS[name]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    rng = np.random.default_rng(seed)
    if name == "M2":                                   # start = M0; kept unless validation improves
        best = start = val_loss(model, val, ab, device)
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        best, start, best_state = float("inf"), float("nan"), None
    best_ep, bad, ep = 0, 0, 0
    for ep in range(1, hp["epochs"] + 1):
        model.train()
        opt.zero_grad()
        tr_loss = []
        for k, i in enumerate(rng.permutation(len(train)), 1):
            c = train[i]
            p = torch.as_tensor(prior(c, ab), dtype=torch.float32, device=device)
            loss = circuit_loss(model(to_batch(c.g, device), p),
                                torch.as_tensor(c.att, dtype=torch.float32, device=device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss on {c.cid}")
            (loss / hp["batch_circuits"]).backward()
            tr_loss.append(loss.item())
            if k % hp["batch_circuits"] == 0 or k == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
        vl = val_loss(model, val, ab, device)
        if vl < best - 1e-7:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
        if ep == 1 or ep % 10 == 0:
            log(f"    {name} seed {seed} epoch {ep:3d}: train {np.mean(tr_loss):.3e}  val {vl:.3e}")
    model.load_state_dict(best_state)
    log(f"  {name} seed {seed}: {ep} epochs, best epoch {best_ep}, val loss {start:.3e} -> {best:.3e}")
    return model, {"seed": seed, "epochs_run": ep, "best_epoch": best_ep, "val_start": start, "val_best": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_v31")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--models", default="M2,M3")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    assert "holdout" not in a.train, "training must not read a holdout"
    assert not os.environ.get("QUEST_UNBLIND"), "unset QUEST_UNBLIND: training never reads holdout3"
    assert os.environ.get("PYTHONHASHSEED") == "0", "export PYTHONHASHSEED=0 first"
    if not a.smoke:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "notes/PREREG_gt_att.md"],
                                 cwd=ROOT, capture_output=True).returncode == 0
        assert tracked, "commit notes/PREREG_gt_att.md before the full training run"
    hp, seeds = dict(HP), [int(s) for s in a.seeds.split(",")]
    out = RUNS / "gt_att" / ("smoke_" + time.strftime("%Y%m%d_%H%M%S") if a.smoke else "main")
    if a.smoke:
        hp.update(epochs=2, patience=2)
        seeds = seeds[:1]
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(out / "log.txt", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    t0 = time.time()
    C = [c for c in load_att_split(a.train, "train") if c.usable_train()]
    train, val = split_train_val(C, SPLIT_SEED)
    ab = fit_prior(train)
    w = fit_linear(train, ab)
    a_min = float(min(c.att.min() for c in train))
    cfg = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": a.smoke, "train_corpus": a.train,
           "n_train": len(train), "n_val": len(val), "train_cids": [c.cid for c in train],
           "val_cids": [c.cid for c in val], "alpha": ab[0], "beta": ab[1], "w": w.tolist(),
           "a_min": a_min, "arch": ARCH, "hp": hp, "seeds": seeds, "models": a.models.split(","),
           "git": git_state(), "python": platform.python_version(), "torch": torch.__version__,
           "numpy": np.__version__, "platform": platform.platform(), "device": str(device),
           "pythonhashseed": os.environ.get("PYTHONHASHSEED")}
    cf = out / "config.json"
    if cf.exists() and json.loads(cf.read_text())["train_cids"] != cfg["train_cids"]:
        raise RuntimeError(f"{cf} exists with a different split; use a new folder")
    cf.write_text(json.dumps(cfg, indent=1))
    log(f"=== train_gt_att {cfg['time']}  smoke={a.smoke}  git {cfg['git']['commit'][:8]}"
        f"{' (DIRTY)' if cfg['git']['dirty'] else ''}  device={device}")
    log(f"circuits: train {len(train)}, validation {len(val)}; prior a = {ab[0]:+.4f} + {ab[1]:+.4f} e; "
        f"a_min {a_min:+.4f}")
    log(f"validation loss  M0 {closed_form_loss(lambda c: predict_m0(c, ab), val):.3e}   "
        f"M1 {closed_form_loss(lambda c: predict_m1(c, ab, w), val):.3e}")
    summary = []
    for name in a.models.split(","):
        for s in seeds:
            ck = out / f"{name}_seed{s}.pt"
            if ck.exists():
                log(f"  {ck.name} exists: not overwritten, skipped")
                continue
            model, info = train_one(name, s, train, val, ab, hp, device, log)
            torch.save({"state_dict": model.state_dict(), "model": name, "arch": ARCH, **info,
                        "alpha": ab[0], "beta": ab[1], "git": cfg["git"]}, ck)
            summary.append({"model": name, **info})
    (out / f"summary_{time.strftime('%H%M%S')}.json").write_text(json.dumps(summary, indent=1))
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
