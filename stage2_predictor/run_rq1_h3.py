"""Holdout3: freeze the primary model (pre-registration Part B), then evaluate ONCE.

--stage freeze   (training data only; holdout3 is never opened)
    1. Calibration control on the width-selection split (train 6-10, judge 12-14):
       gt_size with fragment-layout features vs the same features with
       device-median calibration (placement and SWAP distances kept). Exploratory.
    2. Train the frozen primary gt_size-layout and the control gt_size-layout_uniform
       on all 6-14 qubit circuits (stratified validation for early stopping),
       5 seeds; save the weights and write partB_record.json with SHA-256 hashes.
    quest.sh commits the record and tags it prereg-holdout3-B BEFORE unblinding.

--stage evaluate (requires QUEST_UNBLIND=holdout3; verifies the weight hashes)
    Loads holdout3 once and reports the pre-registered family R:
      R1  primary vs random                         (gap-weighted tau)
      R2  primary vs minimum overhead, METIS, GBM (layout features), layout-ESP
          Holm over the four (ESP added 16 Oct 2026, logged before unblinding)
    Secondary metrics: Kendall tau, top-1 regret, NDCG@3. Exploratory: placement-only
    control, ESP with device-median calibration, results per family and width.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quest_phase2 import baselines as B                              # noqa: E402
from quest_phase2.data import load_split, stratified_split           # noqa: E402
from quest_phase2.metrics import paired_test                         # noqa: E402
from quest_phase2.models import VARIANTS, GraphTransformer           # noqa: E402
from run_rq1 import HP, score_rows, trainable                        # noqa: E402
from run_rq1_v2 import predict, report, standardise_hand, train_variant  # noqa: E402

PRIMARY = ("gt_size", "layout")
CONTROL = ("gt_size", "layout_uniform")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def freeze(a, hp, seeds, device, log):
    out = Path(a.out)
    rows = []
    # 1. calibration control on training data (width selection)
    for agg in ("layout", "layout_uniform"):
        pool = load_split(Path(a.train), "train", aggregation=agg)
        small = [g for g in pool if g.n_qubits <= 10]
        large = [g for g in pool if g.n_qubits >= 12]
        tr, va = stratified_split(small)
        for g in large:
            g.split = "sel_12_14"
        standardise_hand(tr, va + large)
        if agg == "layout":
            rows += score_rows("esp_layout", 0, large, B.score_esp)
            rows += score_rows("esp_uniform", 0, large, B.score_esp_uniform)
        for seed in seeds:
            m = train_variant("gt_size", tr, va, seed, device, hp, log)
            rows += score_rows(f"gt_size-{agg}", seed, large, lambda g: predict(m, g, device))
    (out / "control").mkdir(exist_ok=True)
    report(rows, out / "control", log, ("sel_12_14",))
    control = {f"{x}_vs_{y}": paired_test(rows, x, y, "sel_12_14", "kendall_w")
               for x, y in (("gt_size-layout", "gt_size-layout_uniform"), ("esp_layout", "esp_uniform"))}
    for k, v in control.items():
        log(f"  calibration control {k}: n={v['n']} diff={v['mean_diff']:+.3f} p={v['p']:.2e} r={v['effect_rb']:+.2f}")
    # 2. freeze the primary and the control on all 6-14 qubit circuits
    record = {"primary": "-".join(PRIMARY), "control": "-".join(CONTROL), "seeds": seeds, "hp": hp,
              "git_commit_before": os.environ.get("QUEST_COMMIT", "unknown"),
              "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "weights": {},
              "calibration_control_sel_12_14": control}
    for name, agg in (PRIMARY, CONTROL):
        pool = load_split(Path(a.train), "train", aggregation=agg)
        tr, va = stratified_split(pool)
        standardise_hand(tr, va)
        for seed in seeds:
            m = train_variant(name, tr, va, seed, device, hp, log)
            p = out / f"{name}-{agg}_seed{seed}.pt"
            torch.save(m.state_dict(), p)
            record["weights"][p.name] = sha256(p)
    (out / "partB_record.json").write_text(json.dumps(record, indent=1))
    log(f"PART B FROZEN: {record['primary']} ({len(seeds)} seeds); record written; holdout3 not opened")


def evaluate(a, hp, seeds, device, log):
    assert os.environ.get("QUEST_UNBLIND") == "holdout3", "evaluate needs QUEST_UNBLIND=holdout3"
    out = Path(a.out)
    record = json.loads((out / "partB_record.json").read_text())
    for fname, h in record["weights"].items():
        assert sha256(out / fname) == h, f"weight file {fname} changed after freezing"
    log("weights verified against partB_record.json")
    rows = []
    for name, agg in (PRIMARY, CONTROL):
        pool = load_split(Path(a.train), "train", aggregation=agg)
        tr, va = stratified_split(pool)
        h3 = load_split(Path(a.holdout3), "holdout3", aggregation=agg)
        # only the pre-registered analysis set: first 16 eligible circuits per cell
        chosen = set(json.loads((Path(a.holdout3).parent / "holdout3_selection.json").read_text())["circuits"])
        h3 = [g for g in h3 if g.cid in chosen]
        assert len(h3) == len(chosen), f"{len(h3)} loaded vs {len(chosen)} selected"
        standardise_hand(tr, va + h3)
        if agg == "layout":
            log(f"holdout3 circuits loaded: {len(h3)}")
            for seed in seeds:
                rows += score_rows("random", seed, h3, lambda g: B.score_random(g, seed))
                gbm = B.GBMRanker(seed).fit([g for g in tr if trainable(g)])
                rows += score_rows("gbm-layout", seed, h3, gbm.score)
            rows += score_rows("min_overhead", 0, h3, B.score_min_overhead)
            rows += score_rows("metis", 0, h3, B.score_metis)
            rows += score_rows("esp_layout", 0, h3, B.score_esp)
            rows += score_rows("esp_uniform", 0, h3, B.score_esp_uniform)
        for seed in seeds:
            m = GraphTransformer(hp["d"], hp["heads"], hp["layers"], dropout=hp["dropout"],
                                 **VARIANTS[name]).to(device)
            m.load_state_dict(torch.load(out / f"{name}-{agg}_seed{seed}.pt", map_location=device))
            rows += score_rows(f"{name}-{agg}", seed, h3, lambda g: predict(m, g, device))
    report(rows, out, log, ("holdout3",))
    prim = "-".join(PRIMARY)
    res = {"R1": {"vs_random": paired_test(rows, prim, "random", "holdout3", "kendall_w")}}
    base = ["min_overhead", "metis", "gbm-layout", "esp_layout"]
    r2 = [paired_test(rows, prim, b, "holdout3", "kendall_w") for b in base]
    adj = holm(np.array([r["p"] if np.isfinite(r["p"]) else 1.0 for r in r2]))
    res["R2"] = {b: {**r, "p_holm": float(p)} for b, r, p in zip(base, r2, adj)}
    res["secondary"] = {f"{b}|{m}": paired_test(rows, prim, b, "holdout3", m)
                        for b in ["random"] + base for m in ("kendall", "top1_regret", "ndcg3")}
    res["exploratory"] = {
        "primary_vs_placement_only": paired_test(rows, prim, "-".join(CONTROL), "holdout3", "kendall_w"),
        "esp_vs_esp_placement_only": paired_test(rows, "esp_layout", "esp_uniform", "holdout3", "kendall_w")}
    (out / "holdout3_tests.json").write_text(json.dumps(res, indent=1))
    log("\nPRE-REGISTERED FAMILY R (gap-weighted tau, holdout3)")
    v = res["R1"]["vs_random"]
    log(f"  R1 {prim} vs random: diff={v['mean_diff']:+.3f} p={v['p']:.2e} r={v['effect_rb']:+.2f}")
    for b, v in res["R2"].items():
        log(f"  R2 {prim} vs {b:<13} diff={v['mean_diff']:+.3f} p={v['p']:.2e} Holm={v['p_holm']:.2e} r={v['effect_rb']:+.2f}")
    log("EXPLORATORY")
    for k, v in res["exploratory"].items():
        log(f"  {k}: diff={v['mean_diff']:+.3f} p={v['p']:.2e} r={v['effect_rb']:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["freeze", "evaluate"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout3", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    hp, seeds = dict(HP), [int(s) for s in a.seeds.split(",")]
    if a.smoke:
        hp.update(epochs=2, patience=2); seeds = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(Path(a.out) / "log.txt", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    t0 = time.time()
    log(f"=== holdout3 stage={a.stage} {time.strftime('%Y-%m-%d %H:%M')} device={device} smoke={a.smoke}")
    (freeze if a.stage == "freeze" else evaluate)(a, hp, seeds, device, log)
    log(f"finished in {(time.time() - t0) / 60:.1f} min")
    if a.smoke:
        log("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
