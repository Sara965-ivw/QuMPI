"""RQ1 follow-up: Delta-learning on layout-ESP and a metric-aligned loss.

Why this run exists (21 Oct 2026). On holdout3 the primary model (gt_size-layout)
beat random, minimum overhead and METIS but was not significantly better than
layout-ESP or GBM. Diagnosis on the TRAINING data:
  * the labeller's noise is gate-local (Qiskit NoiseModel.from_backend), so
    log F is, to first order, the sum of per-gate log fidelities that layout-ESP
    computes; the network had to rediscover that sum through pooling;
  * only about 10 per cent of label variance is within circuits, and ListNet on
    within-circuit z-scores stretches nearly tied circuits to unit spread.
Post hoc and exploratory, from holdout3: the model beat ESP on clustered_random and
qaoa_sbm and lost on twolocal, increasingly with width. Because the idea for this
run came partly from holdout3, holdout3 CANNOT confirm it; a fresh holdout4 can.

Design (training data only; holdout3 is never read)
  Split: train 6-10 qubits, early stopping on a 6-10 qubit validation split,
  judged on 12-14 qubits (sel_12_14). holdout2 (seen) reported as exploratory.
  Hardware features: fragment layout (as the frozen primary).
  2 x 2 factorial, 5 seeds per cell, early stopping on gap-weighted tau:
                         ListNet (z-scores)      gap-weighted pairwise
      gt_size            gt_size-listnet         gt_size-pair
      gt_delta (A)       gt_delta-listnet        gt_delta-pair   <- pre-declared candidate
  Reference: gt_ref = the frozen recipe (ListNet, early stopping on plain tau).
  Baselines: random, min overhead, METIS, layout-ESP, GBM-layout, GBM-Delta.

Decision rule (PREREG_delta.md, fixed before this run)
  gt_delta-pair becomes the primary for holdout4 if, on sel_12_14 in gap-weighted
  tau (paired Wilcoxon over circuits, seed-averaged, Holm over the three
  comparisons below, alpha 0.05), it is significantly better than BOTH layout-ESP
  and gt_ref. The comparison with GBM-Delta is reported whatever its outcome: if
  it is not significant, the gain is attributed to the physics prior, not to
  attention. Otherwise the frozen holdout3 result stands as the RQ1 result.
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

from quest_phase2 import baselines as B                                   # noqa: E402
from quest_phase2.data import load_split, stratified_split                # noqa: E402
from quest_phase2.metrics import circuit_metrics, paired_test             # noqa: E402
from quest_phase2.models import (VARIANTS, GraphTransformer, gap_pairwise_loss,  # noqa: E402
                                 listnet_loss, to_batch)
from run_rq1 import HP, score_rows, set_seed, trainable                   # noqa: E402
from run_rq1_hw import holm                                               # noqa: E402
from run_rq1_v2 import predict, report, standardise_hand                  # noqa: E402

AGG = "layout"
# name -> (architecture, loss, early-stopping metric)
NEURAL = {
    "gt_ref":           ("gt_size",  "listnet", "kendall"),
    "gt_size-listnet":  ("gt_size",  "listnet", "kendall_w"),
    "gt_size-pair":     ("gt_size",  "pair",    "kendall_w"),
    "gt_delta-listnet": ("gt_delta", "listnet", "kendall_w"),
    "gt_delta-pair":    ("gt_delta", "pair",    "kendall_w"),
}
CANDIDATE = "gt_delta-pair"
DECISION_VS = ["esp_layout", "gt_ref", "gbm-delta"]      # Holm family; adopt needs the first two


def mean_metric(model, graphs, device, metric):
    v = [circuit_metrics(g.y, predict(model, g, device)).get(metric) for g in graphs if trainable(g)]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float("-inf")


def train(name, train_set, val, seed, device, hp, log):
    arch, loss_kind, stop_metric = NEURAL[name]
    set_seed(seed)
    model = GraphTransformer(hp["d"], hp["heads"], hp["layers"], dropout=hp["dropout"],
                             **VARIANTS[arch]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    data = [g for g in train_set if trainable(g)]
    rng = np.random.default_rng(seed)
    # Delta models start from the untrained state, which ranks exactly as ESP: if training
    # never improves on it, the model honestly falls back to ESP. Other models start at
    # -inf as in the frozen recipe, so gt_ref reproduces run_rq1_v2.train_variant.
    if arch == "gt_delta":
        best = mean_metric(model, val, device, stop_metric)
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        best, best_state = float("-inf"), None
    start, bad, ep = best, 0, 0
    for ep in range(hp["epochs"]):
        model.train()
        opt.zero_grad()
        for k, i in enumerate(rng.permutation(len(data)), 1):
            b = to_batch(data[i], device)
            s = model(b)
            loss = listnet_loss(s, b["y_z"], hp["tau"]) if loss_kind == "listnet" else gap_pairwise_loss(s, b["y"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss on {data[i].cid}")
            (loss / hp["batch_circuits"]).backward()
            if k % hp["batch_circuits"] == 0 or k == len(data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        vm = mean_metric(model, val, device, stop_metric)
        if vm > best + 1e-4:
            best, bad = vm, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    extra = f", esp_w {model.esp_w.item():.2f}" if getattr(model, "delta", False) else ""
    log(f"    {name} seed {seed}: {ep + 1} epochs, val {stop_metric} {start:.3f} -> {best:.3f}{extra}")
    return model


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
    log(f"=== RQ1 Delta {time.strftime('%Y-%m-%d %H:%M')} device={device} smoke={a.smoke}")
    pool = load_split(Path(a.train), "train", aggregation=AGG)
    small = [g for g in pool if g.n_qubits <= 10]
    large = [g for g in pool if g.n_qubits >= 12]
    train_set, val = stratified_split(small)
    for g in large:
        g.split = "sel_12_14"
    hold2 = load_split(Path(a.holdout2), "holdout2_seen", aggregation=AGG) if a.holdout2 else []
    standardise_hand(train_set, val + large + hold2)
    evals = val + large + hold2
    log(f"train {len(train_set)} val {len(val)} sel_12_14 {len(large)} holdout2 (seen, exploratory) {len(hold2)}")

    rows = []
    for seed in seeds:
        rows += score_rows("random", seed, evals, lambda g: B.score_random(g, seed))
    rows += score_rows("min_overhead", 0, evals, B.score_min_overhead)
    rows += score_rows("metis", 0, evals, B.score_metis)
    rows += score_rows("esp_layout", 0, evals, B.score_esp)
    fit = [g for g in train_set if trainable(g)]
    for seed in seeds:
        rows += score_rows("gbm-layout", seed, evals, B.GBMRanker(seed).fit(fit).score)
        gd = B.GBMDeltaRanker(seed).fit(fit)
        rows += score_rows("gbm-delta", seed, evals, gd.score)
    log(f"GBM-Delta ESP scale c = {gd.c:.3f} (log F ~ c * log ESP within circuits)")

    # sanity check: an untrained Delta model must rank exactly as layout-ESP
    m0 = GraphTransformer(hp["d"], hp["heads"], hp["layers"], dropout=hp["dropout"],
                          **VARIANTS["gt_delta"]).to(device)
    gap = max(float(np.max(np.abs(predict(m0, g, device) - g.esp))) for g in evals[:20])
    log(f"untrained gt_delta minus log ESP, max abs: {gap:.2e} (must be 0)")
    assert gap < 1e-5, "Delta model does not start at ESP"

    for name in NEURAL:
        for seed in seeds:
            m = train(name, train_set, val, seed, device, hp, log)
            rows += score_rows(name, seed, evals, lambda g: predict(m, g, device))
    summ = report(rows, out, log, ("sel_12_14", "holdout2_seen"))

    log("\nmodel               split          gap-weighted tau [95% CI]")
    for r in summ:
        if r["split"] in ("sel_12_14", "holdout2_seen"):
            log(f"{r['model']:<19} {r['split']:<14} {r['kendall_w']:+.3f} "
                f"[{r['kendall_w_lo']:+.3f},{r['kendall_w_hi']:+.3f}]")

    # factorial effects (exploratory): A = Delta, B = loss
    tests = {}
    pairs = {"A_delta|listnet": ("gt_delta-listnet", "gt_size-listnet"),
             "A_delta|pair": ("gt_delta-pair", "gt_size-pair"),
             "B_loss|gt_size": ("gt_size-pair", "gt_size-listnet"),
             "B_loss|gt_delta": ("gt_delta-pair", "gt_delta-listnet"),
             "gbm-delta_vs_esp": ("gbm-delta", "esp_layout"),
             "gt_ref_vs_esp": ("gt_ref", "esp_layout")}
    log("\nfactorial and reference comparisons (exploratory), sel_12_14, gap-weighted tau")
    for k, (x, y) in pairs.items():
        tests[k] = paired_test(rows, x, y, "sel_12_14", "kendall_w")
        v = tests[k]
        log(f"  {k:<20} {x} vs {y}: n={v['n']} diff={v['mean_diff']:+.3f} p={v['p']:.2e} r={v['effect_rb']:+.2f}")

    # pre-declared decision
    res = [paired_test(rows, CANDIDATE, b, "sel_12_14", "kendall_w") for b in DECISION_VS]
    ps = np.array([r["p"] if np.isfinite(r["p"]) else 1.0 for r in res])
    adj = holm(ps)
    better = {b: bool(p < 0.05 and r["mean_diff"] > 0) for b, r, p in zip(DECISION_VS, res, adj)}
    adopt = better["esp_layout"] and better["gt_ref"]
    decision = {"rule": "adopt gt_delta-pair for holdout4 if significantly better than layout-ESP AND gt_ref "
                        "on sel_12_14 in gap-weighted tau (Wilcoxon, Holm over 3, alpha 0.05)",
                "tests": {b: {**r, "p_holm": float(p)} for b, r, p in zip(DECISION_VS, res, adj)},
                "significantly_better_than": better, "adopt": adopt}
    tests["decision"] = decision
    (out / "tests.json").write_text(json.dumps(tests, indent=1))
    log(f"\nDECISION vs {DECISION_VS}:")
    for b, r, p in zip(DECISION_VS, res, adj):
        log(f"  {CANDIDATE} vs {b:<11} diff={r['mean_diff']:+.3f}  Holm p={p:.2e}  better={better[b]}")
    log(f"ADOPT {CANDIDATE} FOR HOLDOUT4: {adopt}")
    (out / "config.json").write_text(json.dumps({
        "hp": hp, "seeds": seeds, "aggregation": AGG, "neural": NEURAL, "candidate": CANDIDATE,
        "device": str(device), "torch": torch.__version__, "python": platform.python_version(),
        "git_commit": os.environ.get("QUEST_COMMIT", "unknown"), "smoke": a.smoke}, indent=1))
    log(f"finished in {(time.time() - t0) / 60:.1f} min -> {out}")
    if a.smoke:
        log("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
