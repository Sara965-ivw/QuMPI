"""RQ1 hardware question and holdout3 model selection (training data only).

Question. Does calibration information help rank cuts once it describes the
qubits each fragment actually uses (fragment layout) instead of the uncut
circuit's qubits? And which model is frozen as the primary for holdout3
(pre-registration Part B)?

Design (width-selection protocol, as in run_rq1_v2: no holdout3 circuit is read)
  train 6-10 qubits, early stopping on a 6-10 qubit validation split,
  judge on 12-14 qubits (sel_12_14). holdout2 (seen) is reported as exploratory.
  Models x hardware configuration, 5 seeds each:
      gt_size, gt_hybrid  x  {uncut (as before), layout (new), none}
  Baselines: random, minimum overhead, METIS, layout-ESP, GBM per configuration.
  Metrics: gap-weighted Kendall tau (primary for holdout3), Kendall tau,
  top-1 regret, NDCG@3.

Part B selection rule (written into PREREG_holdout3.md before this runs):
  the current primary (gt_size, uncut) is kept unless an alternative among
  {gt_size-layout, gt_hybrid-uncut, gt_hybrid-layout} is significantly better
  on sel_12_14 in gap-weighted tau (paired Wilcoxon over circuits, seed-averaged,
  Holm over the three alternatives, alpha 0.05); if several qualify, the one with
  the highest mean is chosen.
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

from quest_phase2 import baselines as B                              # noqa: E402
from quest_phase2.data import load_split, stratified_split           # noqa: E402
from quest_phase2.metrics import paired_test                         # noqa: E402
from run_rq1 import HP, score_rows, trainable                        # noqa: E402
from run_rq1_v2 import predict, report, standardise_hand, train_variant  # noqa: E402

CONFIGS = {"uncut": "both", "layout": "layout", "none": "none"}
MODELS = ["gt_size", "gt_hybrid"]
PRIMARY = "gt_size-uncut"
ALTERNATIVES = ["gt_size-layout", "gt_hybrid-uncut", "gt_hybrid-layout"]


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout2", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    assert "holdout3" not in str(a.train) + str(a.holdout2), "holdout3 stays blinded"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hp, seeds = dict(HP), [int(s) for s in a.seeds.split(",")]
    if a.smoke:
        hp.update(epochs=2, patience=2); seeds = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(out / "log.txt", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    t0 = time.time()
    log(f"=== RQ1 hardware / Part B selection {time.strftime('%Y-%m-%d %H:%M')} device={device} smoke={a.smoke}")
    rows = []
    for cfg, agg in CONFIGS.items():
        pool = load_split(Path(a.train), "train", aggregation=agg)
        small = [g for g in pool if g.n_qubits <= 10]
        large = [g for g in pool if g.n_qubits >= 12]
        train, val = stratified_split(small)
        for g in large:
            g.split = "sel_12_14"
        hold2 = load_split(Path(a.holdout2), "holdout2_seen", aggregation=agg) if a.holdout2 else []
        standardise_hand(train, val + large + hold2)
        evals = val + large + hold2
        log(f"[{cfg}] train {len(train)} val {len(val)} sel_12_14 {len(large)} holdout2 (seen, exploratory) {len(hold2)}")
        if cfg == "uncut":     # structure-only baselines do not depend on the configuration
            for seed in seeds:
                rows += score_rows("random", seed, evals, lambda g: B.score_random(g, seed))
            rows += score_rows("min_overhead", 0, evals, B.score_min_overhead)
            rows += score_rows("metis", 0, evals, B.score_metis)
            rows += score_rows("esp_layout", 0, evals, B.score_esp)
        for seed in seeds:
            gbm = B.GBMRanker(seed).fit([g for g in train if trainable(g)])
            rows += score_rows(f"gbm-{cfg}", seed, evals, gbm.score)
        for name in MODELS:
            for seed in seeds:
                m = train_variant(name, train, val, seed, device, hp, log)
                rows += score_rows(f"{name}-{cfg}", seed, evals, lambda g: predict(m, g, device))
    summ = report(rows, out, log, ("val", "sel_12_14", "holdout2_seen"))

    log("\nmodel               split          gap-weighted tau [95% CI]")
    for r in summ:
        if r["split"] in ("sel_12_14", "holdout2_seen"):
            log(f"{r['model']:<19} {r['split']:<14} {r['kendall_w']:+.3f} "
                f"[{r['kendall_w_lo']:+.3f},{r['kendall_w_hi']:+.3f}]")

    tests = {}
    for split in ("sel_12_14", "holdout2_seen"):
        for name in MODELS + ["gbm"]:
            for other in ("none", "uncut"):
                for metric in ("kendall_w", "kendall"):
                    tests[f"{split}|{name}-layout_vs_{name}-{other}|{metric}"] = paired_test(
                        rows, f"{name}-layout", f"{name}-{other}", split, metric)
        for base in ("min_overhead", "metis", "esp_layout", "random"):
            tests[f"{split}|esp_layout_vs_{base}|kendall_w"] = paired_test(
                rows, "esp_layout", base, split, "kendall_w") if base != "esp_layout" else {}
    (out / "tests.json").write_text(json.dumps(tests, indent=1))
    log("\nhardware question (layout vs none / uncut), paired Wilcoxon over circuits")
    for k, v in tests.items():
        if v and "_vs_" in k and "esp" not in k:
            log(f"  {k:<62} n={v['n']:<3} diff={v['mean_diff']:+.3f} p={v['p']:.2e} r={v['effect_rb']:+.2f}")

    # Part B selection rule (pre-registered)
    res = [paired_test(rows, alt, PRIMARY, "sel_12_14", "kendall_w") for alt in ALTERNATIVES]
    ps = np.array([r["p"] if np.isfinite(r["p"]) else 1.0 for r in res])
    adj = holm(ps)
    qual = [alt for alt, r, p in zip(ALTERNATIVES, res, adj) if p < 0.05 and r["mean_diff"] > 0]
    sel = {r["model"]: r for r in summ if r["split"] == "sel_12_14"}
    chosen = max(qual, key=lambda n: sel[n]["kendall_w"]) if qual else PRIMARY
    decision = {"rule": "keep gt_size-uncut unless an alternative is significantly better on sel_12_14 "
                        "in gap-weighted tau (Wilcoxon, Holm over 3, alpha 0.05)",
                "tests": {alt: {**r, "p_holm": float(p)} for alt, r, p in zip(ALTERNATIVES, res, adj)},
                "qualifying": qual, "selected": chosen,
                "kendall_w_sel": {m: sel[m]["kendall_w"] for m in [PRIMARY] + ALTERNATIVES if m in sel}}
    (out / "partB_selection.json").write_text(json.dumps(decision, indent=1))
    log(f"\nPART B SELECTED PRIMARY MODEL: {chosen}  (qualifying alternatives: {qual or 'none'})")
    (out / "config.json").write_text(json.dumps({
        "hp": hp, "seeds": seeds, "configs": CONFIGS, "models": MODELS, "device": str(device),
        "torch": torch.__version__, "python": platform.python_version(),
        "git_commit": os.environ.get("QUEST_COMMIT", "unknown"), "smoke": a.smoke}, indent=1))
    log(f"finished in {(time.time() - t0) / 60:.1f} min -> {out}")
    if a.smoke:
        log("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
