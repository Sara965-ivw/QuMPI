"""Card C, part 1: a fidelity-predicting head and the MAE of predicted fidelity
(proposal line 563: "Mean absolute error of predicted fidelity against ground truth
where computable, rank correlation between predicted and true cut ordering, and the
degradation of both as a function of qubit count").

Decisions (22 Oct 2026): C1 (b) the model predicts fidelity (proposal line 461);
C1-b (2) ranking and regression losses together, so that predictions are fidelity
values. The frozen RQ1 model ranks with a scale-free score and has no MAE.

Model: gt_size architecture with fragment-layout features (as the frozen model).
Its output F_hat is a fidelity. Loss per circuit (both terms dimensionless, so
neither dominates by its units):
    L = L_rank(F_hat / T) + mean_c (F_hat_c - F_c)^2 / Var_train(F)
  * L_rank: gap-weighted pairwise logistic loss (models.gap_pairwise_loss), the loss
    that improved ranking on 21 Oct (+0.06 to +0.085 gap-weighted tau);
  * T: median within-circuit |F_i - F_j| over training pairs, which puts typical
    candidate differences on the logistic loss's natural scale;
  * Var_train(F): variance of all training labels.
The output bias starts at the mean training fidelity. Early stopping on the same
loss on the validation split (6-10 q). 5 seeds.

Baselines for MAE (each fitted on training circuits only):
  constant   mean training fidelity
  esp_lin    F = a + b * log ESP (least squares)
  gbm_reg    gradient boosting regression on the hand features (which include the
             fragment log-ESP), same hyperparameters as GBMRanker, target F
Splits: val (6-10 q), sel_12_14 (12-14 q), holdout2 (seen, exploratory). holdout3 is
not read. Tests (stated before the run): per-circuit MAE on sel_12_14, model vs each
baseline, paired Wilcoxon over circuits (seed-averaged), Holm over the three.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quest_phase2.data import load_split, stratified_split                 # noqa: E402
from quest_phase2.metrics import bootstrap_ci, kendall_gap_weighted, paired_test  # noqa: E402
from quest_phase2.models import VARIANTS, GraphTransformer, gap_pairwise_loss, to_batch  # noqa: E402
from run_rq1 import HP, set_seed, trainable, write_csv                     # noqa: E402
from run_rq1_hw import holm                                                # noqa: E402
from run_rq1_v2 import predict, standardise_hand                           # noqa: E402

BASELINES = ["constant", "esp_lin", "gbm_reg"]


def loss_fn(fhat, y, T, var):
    return gap_pairwise_loss(fhat / T, y) + torch.mean((fhat - y) ** 2) / var


def val_loss(model, graphs, device, T, var):
    model.eval()
    tot = []
    with torch.no_grad():
        for g in graphs:
            if trainable(g):
                b = to_batch(g, device)
                tot.append(float(loss_fn(model(b), b["y"], T, var)))
    return float(np.mean(tot)) if tot else float("inf")


def train(train_set, val, seed, device, hp, T, var, f_mean, log):
    set_seed(seed)
    model = GraphTransformer(hp["d"], hp["heads"], hp["layers"], dropout=hp["dropout"],
                             **VARIANTS["gt_size"]).to(device)
    with torch.no_grad():
        model.out.head[-1].bias.fill_(f_mean)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    data = [g for g in train_set if trainable(g)]
    rng = np.random.default_rng(seed)
    best, best_state, bad, ep = float("inf"), None, 0, 0
    for ep in range(hp["epochs"]):
        model.train()
        opt.zero_grad()
        for k, i in enumerate(rng.permutation(len(data)), 1):
            b = to_batch(data[i], device)
            loss = loss_fn(model(b), b["y"], T, var)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss on {data[i].cid}")
            (loss / hp["batch_circuits"]).backward()
            if k % hp["batch_circuits"] == 0 or k == len(data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        vl = val_loss(model, val, device, T, var)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    log(f"    gt_size-fid seed {seed}: {ep + 1} epochs, best validation loss {best:.4f}")
    return model


def rows_for(name, seed, graphs, predictor):
    out = []
    for g in graphs:
        f = np.asarray(predictor(g), float)
        y = np.asarray(g.y, float)
        out.append({"model": name, "seed": seed, "split": g.split, "cid": g.cid, "family": g.family,
                    "n_qubits": g.n_qubits, "mae": float(np.mean(np.abs(f - y))),
                    "rmse": float(np.sqrt(np.mean((f - y) ** 2))),
                    "kendall_w": kendall_gap_weighted(y, f) if len(y) > 1 and np.ptp(y) > 0 else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout2", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    assert "holdout3" not in str(a.train) + str(a.holdout2), "holdout3 is not used by this run"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hp, seeds = dict(HP), [int(s) for s in a.seeds.split(",")]
    if a.smoke:
        hp.update(epochs=2, patience=2); seeds = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(out / "log.txt", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    t0 = time.time()
    log(f"=== Card C part 1: fidelity head and MAE {time.strftime('%Y-%m-%d %H:%M')} device={device} smoke={a.smoke}")
    pool = load_split(Path(a.train), "train", aggregation="layout")
    small = [g for g in pool if g.n_qubits <= 10]
    large = [g for g in pool if g.n_qubits >= 12]
    train_set, val = stratified_split(small)
    for g in large:
        g.split = "sel_12_14"
    hold2 = load_split(Path(a.holdout2), "holdout2_seen", aggregation="layout") if a.holdout2 else []
    standardise_hand(train_set, val + large + hold2)
    evals = val + large + hold2
    fit = [g for g in train_set if trainable(g)]
    y_all = np.concatenate([g.y for g in fit])
    diffs = np.concatenate([np.abs(g.y[:, None] - g.y[None, :])[np.triu_indices(len(g.y), 1)] for g in fit])
    T = float(np.median(diffs[diffs > 0]))
    var = float(np.var(y_all))
    f_mean = float(np.mean(y_all))
    log(f"train {len(train_set)} val {len(val)} sel_12_14 {len(large)} holdout2 (seen) {len(hold2)}; "
        f"T = {T:.5f}, Var(F) = {var:.6f}, mean F = {f_mean:.4f}")

    rows = rows_for("constant", 0, evals, lambda g: np.full(g.n_cand, f_mean))
    X = np.concatenate([np.asarray(g.esp, float) for g in fit])
    b, a0 = np.polyfit(X, y_all, 1)
    log(f"esp_lin: F = {a0:.4f} + {b:.4f} * log ESP")
    rows += rows_for("esp_lin", 0, evals, lambda g: a0 + b * np.asarray(g.esp, float))
    from sklearn.ensemble import HistGradientBoostingRegressor
    for seed in seeds:
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                            l2_regularization=1e-3, random_state=seed)
        gbm.fit(np.concatenate([g.hand for g in fit]), y_all)
        rows += rows_for("gbm_reg", seed, evals, lambda g: gbm.predict(g.hand))
    for seed in seeds:
        m = train(train_set, val, seed, device, hp, T, var, f_mean, log)
        rows += rows_for("gt_size-fid", seed, evals, lambda g: predict(m, g, device))
    write_csv(out / "per_circuit.csv", rows)

    def summary(group_keys):
        groups = {}
        for r in rows:
            groups.setdefault(tuple(r[k] for k in group_keys), {}).setdefault(r["cid"], []).append(r)
        res = []
        for key, by_c in sorted(groups.items()):
            rec = dict(zip(group_keys, key)); rec["n_circuits"] = len(by_c)
            for met in ("mae", "rmse", "kendall_w"):
                vals = [np.mean([x[met] for x in rs if x[met] is not None]) for rs in by_c.values()
                        if any(x[met] is not None for x in rs)]
                rec[met], rec[met + "_lo"], rec[met + "_hi"] = bootstrap_ci(vals)
            res.append(rec)
        return res

    summ = summary(("model", "split"))
    write_csv(out / "summary.csv", summ)
    write_csv(out / "summary_by_width.csv", summary(("model", "split", "n_qubits")))
    log("\nmodel         split          MAE [95% CI]                  RMSE     gap-weighted tau")
    for r in summ:
        log(f"{r['model']:<13} {r['split']:<14} {r['mae']:.4f} [{r['mae_lo']:.4f},{r['mae_hi']:.4f}]   "
            f"{r['rmse']:.4f}   {r['kendall_w']:+.3f}")
    log("\nMAE by width (degradation, proposal line 564)")
    for r in summary(("model", "n_qubits")):
        log(f"  {r['model']:<13} n={r['n_qubits']:<3} MAE {r['mae']:.4f} [{r['mae_lo']:.4f},{r['mae_hi']:.4f}]")

    # paired MAE tests (lower is better: diff = baseline - model > 0 means the model is better)
    res = []
    for base in BASELINES:
        t = paired_test(rows, base, "gt_size-fid", "sel_12_14", "mae")
        res.append(t)
    adj = holm(np.array([t["p"] if np.isfinite(t["p"]) else 1.0 for t in res]))
    tests = {f"gt_size-fid_vs_{bname}": {**t, "p_holm": float(p), "note": "mean_diff = baseline MAE - model MAE"}
             for bname, t, p in zip(BASELINES, res, adj)}
    (out / "tests.json").write_text(json.dumps(tests, indent=1))
    log("\nMAE on sel_12_14, paired Wilcoxon (diff = baseline MAE - model MAE; > 0 favours the model), Holm over 3")
    for bname, t, p in zip(BASELINES, res, adj):
        log(f"  vs {bname:<9} n={t['n']} diff={t['mean_diff']:+.4f} p={t['p']:.2e} Holm={p:.2e} r={t['effect_rb']:+.2f}")
    (out / "config.json").write_text(json.dumps({
        "hp": hp, "seeds": seeds, "T": T, "var_F": var, "mean_F": f_mean, "esp_lin": [a0, b],
        "device": str(device), "torch": torch.__version__, "python": platform.python_version(),
        "git_commit": os.environ.get("QUEST_COMMIT", "unknown"), "smoke": a.smoke}, indent=1))
    log(f"finished in {(time.time() - t0) / 60:.1f} min -> {out}")
    if a.smoke:
        log("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
