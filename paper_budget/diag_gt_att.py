"""Stage-1 diagnostics for the attenuation Transformer, training data only (notes/PREDICTIONS_diag_gt.md).

Why the Transformer did not beat M1 on holdout3 (notes/RESULTS_gt_att_holdout3.md): too little data,
a calibration it did not learn, or a loss that does not match the decision? This script answers the
first two without touching any holdout:

  D1 learning curve   train on 25%, 50% and 100% of each fold's training circuits;
  D2 calibration      M2b = M1 + Transformer residual (the network starts from the linear calibration).

Design: 5-fold cross-validation over the 169 usable train_v31 circuits, grouped by circuit and
stratified by family (StratifiedGroupKFold, seed 2026). In each fold, 20% of the (randomly ordered) training
circuits, at least 5, are held back for early stopping; the per-cell stratified split of training is
not used because the 25% subsets leave most (family, width) cells with fewer than 3 circuits. Out-of-fold predictions of all circuits are pooled per model and fraction;
Transformer predictions are averaged over 3 seeds. Frozen RQ1 hyperparameters (no tuning here).

  python -m paper_budget.diag_gt_att            (about 1 hour on the RTX 5070)
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from paper_budget.att_data import load_att_split
from paper_budget.config import RUNS, SPLIT_SEED
from paper_budget.eval_gt_att import boot_ratio, fit_metrics, gm, log_regret
from paper_budget.models_att import (DirectGT, ResidualGT, circuit_loss, clip_att, fit_linear,
                                     fit_prior, predict_m0, predict_m1)
from quest_phase2.models import to_batch  # noqa: E402  (frozen)
from run_rq1 import HP, set_seed  # noqa: E402  (frozen)

FRACTIONS = (0.25, 0.5, 1.0)
SEEDS = (0, 1, 2)
NN = ("M2", "M2b", "M3")
OUT = RUNS / "diag_gt"


def _t(x, device):
    return torch.as_tensor(np.asarray(x, float), dtype=torch.float32, device=device)


def train_nn(kind, seed, tr, va, base, hp, device):
    """base(c) -> starting prediction (M0 for M2, M1 for M2b; unused for M3)."""
    set_seed(seed)
    model = (DirectGT if kind == "M3" else ResidualGT)().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    rng = np.random.default_rng(seed)

    def vloss():
        model.eval()
        with torch.no_grad():
            return float(np.mean([float(circuit_loss(model(to_batch(c.g, device), _t(base(c), device)),
                                                     _t(c.att, device))) for c in va]))

    best = vloss() if kind != "M3" else float("inf")
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0
    for _ in range(hp["epochs"]):
        model.train()
        opt.zero_grad()
        for k, i in enumerate(rng.permutation(len(tr)), 1):
            c = tr[i]
            loss = circuit_loss(model(to_batch(c.g, device), _t(base(c), device)), _t(c.att, device))
            (loss / hp["batch_circuits"]).backward()
            if k % hp["batch_circuits"] == 0 or k == len(tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
        vl = vloss()
        if vl < best - 1e-7:
            best, bad = vl, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    assert not os.environ.get("QUEST_UNBLIND"), "diagnostics use training data only: unset QUEST_UNBLIND"
    assert os.environ.get("PYTHONHASHSEED") == "0", "export PYTHONHASHSEED=0 first"
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "diag.json"
    if out.exists():
        raise FileExistsError(f"{out} exists: the diagnostic is run once")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    C = [c for c in load_att_split("train_v31", "train") if c.usable_train() and c.usable_eval()]
    fam = np.array([c.g.family for c in C])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
                 .split(np.zeros(len(C)), fam, groups=[c.cid for c in C]))
    print(f"circuits: {len(C)} (usable for training and evaluation); 5 folds; device {device}", flush=True)
    models = ("M0", "M1") + NN
    pred = {(m, f): [None] * len(C) for m in models for f in FRACTIONS}
    for k, (tr_idx, te_idx) in enumerate(folds):
        order = np.random.default_rng([SPLIT_SEED, k]).permutation(tr_idx)   # nested subsets
        for f in FRACTIONS:
            sub = [C[i] for i in order[: max(int(round(f * len(order))), 10)]]
            n_va = max(5, int(round(0.2 * len(sub))))            # random circuit-level early-stop set
            va, tr = sub[:n_va], sub[n_va:]                        # sub is already in random order
            ab = fit_prior(tr)
            w = fit_linear(tr, ab)
            a_min = float(min(c.att.min() for c in tr))
            base = {"M2": lambda c: predict_m0(c, ab), "M2b": lambda c: predict_m1(c, ab, w),
                    "M3": lambda c: np.zeros(c.n_cand)}
            for i in te_idx:
                pred[("M0", f)][i] = clip_att(predict_m0(C[i], ab), a_min)[0]
                pred[("M1", f)][i] = clip_att(predict_m1(C[i], ab, w), a_min)[0]
            for m in NN:
                acc = {i: [] for i in te_idx}
                for s in SEEDS:
                    net = train_nn(m, s, tr, va, base[m], HP, device)
                    with torch.no_grad():
                        for i in te_idx:
                            c = C[i]
                            acc[i].append(net(to_batch(c.g, device), _t(base[m](c), device)).cpu().numpy())
                for i in te_idx:
                    pred[(m, f)][i] = clip_att(np.mean(acc[i], axis=0).astype(float), a_min)[0]
            print(f"fold {k + 1}/5, fraction {f:.2f}: {len(tr)} train / {len(va)} early-stop circuits, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)

    with open(OUT / "predictions.pkl", "wb") as fh:                 # saved before any metric is computed
        pickle.dump({"cids": [c.cid for c in C], "folds": [te.tolist() for _, te in folds],
                     "pred": {f"{m}|{f}": [q.tolist() for q in P] for (m, f), P in pred.items()}}, fh)
    res = {"n_circuits": len(C), "fractions": FRACTIONS, "seeds": SEEDS, "hp": HP, "results": {}}
    L = {}
    for m in models:
        for f in FRACTIONS:
            P = pred[(m, f)]
            L[(m, f)] = np.array([log_regret(c, q) for c, q in zip(C, P)])
            res["results"][f"{m}|{f}"] = {"fit": fit_metrics(C, P), "regret_gm": gm(L[(m, f)]).tolist()}
    for ref in ("M1", "M2"):
        r = boot_ratio(L[("M2b", 1.0)], L[(ref, 1.0)])
        res["results"][f"M2b_vs_{ref}_ci95"] = r.tolist()
    out.write_text(json.dumps(res, indent=1))

    print("\nD1/D2: out-of-fold within-circuit R^2 | Kendall tau_b | regret at 1e8 | regret at 1e9")
    print(f"{'model':<6}" + "".join(f"{'frac ' + str(f):>40}" for f in FRACTIONS))
    for m in models:
        row = f"{m:<6}"
        for f in FRACTIONS:
            x = res["results"][f"{m}|{f}"]
            row += (f"{x['fit']['r2_within']:>+11.3f}{x['fit']['kendall_tau_b']:>9.3f}"
                    f"{x['regret_gm'][5]:>10.4f}{x['regret_gm'][6]:>10.4f}")
        print(row)
    ci = res["results"]["M2b_vs_M1_ci95"]
    print(f"\nM2b / M1 regret at 1e8: 95% CI [{ci[0][5]:.4f}, {ci[1][5]:.4f}]; "
          f"at 1e9: [{ci[0][6]:.4f}, {ci[1][6]:.4f}]  (all data)")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
