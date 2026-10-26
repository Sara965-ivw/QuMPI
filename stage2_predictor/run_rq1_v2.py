"""RQ1, second stage: size-aware model selection without the test set, then one
confirmatory evaluation on a FRESH holdout.

Why: run 1 (tag rq1-run1) generalised well within 6-14 qubits but not to the
16-20 qubit holdout. The original holdout has now been seen, so it cannot be
used to choose a model. This script therefore separates the two steps.

  --protocol widthsel   train on 6-10 qubits, select on 12-14 qubits (a stand-in
                        size-extrapolation test built only from training data).
                        Compares the pre-declared variants in models.VARIANTS
                        plus GBM. Selection rule, fixed in advance: highest mean
                        Kendall on 12-14 qubits; if the best two overlap in
                        their 95 per cent intervals, the simpler variant wins
                        (order: gt_run1 < gt_sum < gt_size < gt_hybrid).
  --protocol final      train the selected variant on all 6-14 qubit circuits
                        (train/val split as run 1), evaluate ONCE on the fresh
                        holdout (holdout2, new seeds), report the original
                        holdout as secondary, and run the hardware ablations
                        (aggregation none / min / mean) for the selected variant.
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

from quest_phase2 import baselines as B                       # noqa: E402
from quest_phase2.data import load_split, stratified_split    # noqa: E402
from quest_phase2.metrics import paired_test, summarise       # noqa: E402
from quest_phase2.models import GIN, VARIANTS, GraphTransformer, listnet_loss, to_batch  # noqa: E402
from run_rq1 import HP, score_rows, set_seed, trainable, write_csv  # noqa: E402

ORDER = ["gt_run1", "gt_sum", "gt_size", "gt_hybrid"]


def standardise_hand(train, others):
    X = np.concatenate([g.hand for g in train])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    for g in train + others:
        g.hand_z = ((g.hand - mu) / sd).astype(np.float32)


@torch.no_grad()
def predict(model, g, device):
    model.eval()
    return model(to_batch(g, device)).float().cpu().numpy()


def mean_kendall(model, graphs, device):
    from quest_phase2.metrics import circuit_metrics
    v = [circuit_metrics(g.y, predict(model, g, device)).get("kendall") for g in graphs if trainable(g)]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float("-inf")


def train_variant(name, train, sel, seed, device, hp, log):
    set_seed(seed)
    if name == "gin":
        model = GIN(hp["d"], hp["layers"], dropout=hp["dropout"])
    else:
        model = GraphTransformer(hp["d"], hp["heads"], hp["layers"], dropout=hp["dropout"],
                                 **VARIANTS[name])
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    data = [g for g in train if trainable(g)]
    rng = np.random.default_rng(seed)
    best, best_state, bad, ep = float("-inf"), None, 0, 0
    for ep in range(hp["epochs"]):
        model.train()
        opt.zero_grad()
        for k, i in enumerate(rng.permutation(len(data)), 1):
            b = to_batch(data[i], device)
            loss = listnet_loss(model(b), b["y_z"], hp["tau"]) / hp["batch_circuits"]
            loss.backward()
            if k % hp["batch_circuits"] == 0 or k == len(data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        vk = mean_kendall(model, sel, device)
        if vk > best + 1e-4:
            best, bad = vk, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    log(f"    {name} seed {seed}: {ep + 1} epochs, best early-stop Kendall {best:.3f}")
    return model


def report(rows, out, log, splits):
    write_csv(out / "per_circuit.csv", rows)
    summ = summarise(rows, ("model", "split"))
    write_csv(out / "summary.csv", summ)
    write_csv(out / "summary_by_width.csv", summarise(rows, ("model", "split", "n_qubits")))
    write_csv(out / "summary_by_family.csv", summarise(rows, ("model", "split", "family")))
    log("\nmodel               split          circuits  Kendall [95% CI]         top-1 regret  top-1 hit")
    for r in summ:
        if r["split"] in splits:
            log(f"{r['model']:<19} {r['split']:<14} {r['n_circuits']:>6}  "
                f"{r['kendall']:+.3f} [{r['kendall_lo']:+.3f},{r['kendall_hi']:+.3f}]   "
                f"{r['top1_regret']:.4f}        {r['top1_hit']:.2f}")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["widthsel", "final"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--holdout2", default=None)
    ap.add_argument("--variant", default=None, help="final protocol: the selected variant")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hp, seeds = dict(HP), [int(s) for s in a.seeds.split(",")]
    lim = None
    if a.smoke:
        hp.update(epochs=2, patience=2)
        seeds = [0]
        lim = None if a.protocol == "widthsel" else 30   # widthsel needs every width present
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(out / "log.txt", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    t0 = time.time()
    log(f"=== RQ1 v2 protocol={a.protocol} {time.strftime('%Y-%m-%d %H:%M')} device={device} smoke={a.smoke}")
    rows = []
    if a.protocol == "widthsel":
        pool = load_split(Path(a.train), "train", limit=lim)
        small = [g for g in pool if g.n_qubits <= 10]
        large = [g for g in pool if g.n_qubits >= 12]
        train, val = stratified_split(small)
        for g in large:
            g.split = "sel_12_14"
        standardise_hand(train, val + large)
        log(f"train {len(train)} (6-10q)  val {len(val)}  selection set {len(large)} (12-14q)")
        evals = val + large
        for seed in seeds:
            rows += score_rows("random", seed, evals, lambda g: B.score_random(g, seed))
            gbm = B.GBMRanker(seed).fit([g for g in train if trainable(g)])
            rows += score_rows("gbm", seed, evals, gbm.score)
        rows += score_rows("min_overhead", 0, evals, B.score_min_overhead)
        rows += score_rows("metis", 0, evals, B.score_metis)
        for name in ORDER + ["gin"]:
            log(name)
            for seed in seeds:
                # early stopping on the 6-10q validation split: the 12-14q set stays a test
                m = train_variant(name, train, val, seed, device, hp, log)
                rows += score_rows(name, seed, evals, lambda g: predict(m, g, device))
        summ = report(rows, out, log, ("val", "sel_12_14"))
        sel = {r["model"]: r for r in summ if r["split"] == "sel_12_14" and r["model"] in ORDER}
        ranked = sorted(ORDER, key=lambda n: -sel[n]["kendall"])
        best, second = ranked[0], ranked[1]
        overlap = sel[second]["kendall_hi"] >= sel[best]["kendall_lo"]
        chosen = min(best, second, key=ORDER.index) if overlap else best
        decision = {"ranking": ranked, "best": best, "second": second, "ci_overlap": overlap,
                    "selected": chosen,
                    "kendall_12_14": {n: sel[n]["kendall"] for n in ORDER}}
        (out / "selection.json").write_text(json.dumps(decision, indent=1))
        log(f"\nSELECTED VARIANT: {chosen}  (best {best}, overlap with {second}: {overlap})")
    else:
        assert a.variant in VARIANTS, "pass --variant from selection.json"
        pool = load_split(Path(a.train), "train", limit=lim)
        train, val = stratified_split(pool)
        hold2 = load_split(Path(a.holdout2), "holdout2", limit=lim) if a.holdout2 else []
        hold1 = load_split(Path(a.holdout), "holdout", limit=lim) if a.holdout else []
        standardise_hand(train, val + hold1 + hold2)
        log(f"train {len(train)} val {len(val)} holdout2 (fresh) {len(hold2)} holdout (seen) {len(hold1)}")
        evals = val + hold2 + hold1
        for seed in seeds:
            rows += score_rows("random", seed, evals, lambda g: B.score_random(g, seed))
            gbm = B.GBMRanker(seed).fit([g for g in train if trainable(g)])
            rows += score_rows("gbm", seed, evals, gbm.score)
        rows += score_rows("min_overhead", 0, evals, B.score_min_overhead)
        rows += score_rows("metis", 0, evals, B.score_metis)
        for seed in seeds:
            m = train_variant(a.variant, train, val, seed, device, hp, log)
            rows += score_rows(a.variant, seed, evals, lambda g: predict(m, g, device))
            if not a.smoke:
                torch.save(m.state_dict(), out / f"{a.variant}_seed{seed}.pt")
        # hardware ablations: same variant, hardware features removed or aggregated one way
        for agg in ("none", "min", "mean"):
            log(f"ablation aggregation={agg}")
            p2 = load_split(Path(a.train), "train", limit=lim, aggregation=agg)
            tr2, va2 = stratified_split(p2)
            h2 = load_split(Path(a.holdout2), "holdout2", limit=lim, aggregation=agg) if a.holdout2 else []
            standardise_hand(tr2, va2 + h2)
            for seed in seeds:
                m = train_variant(a.variant, tr2, va2, seed, device, hp, log)
                rows += score_rows(f"{a.variant}_hw-{agg}", seed, va2 + h2,
                                   lambda g: predict(m, g, device))
        summ = report(rows, out, log, ("val", "holdout2", "holdout"))
        tests = {}
        for split in ("val", "holdout2", "holdout"):
            for base in ("random", "min_overhead", "metis", "gbm", f"{a.variant}_hw-none"):
                for metric in ("kendall", "top1_regret"):
                    tests[f"{split}|{a.variant}_vs_{base}|{metric}"] = paired_test(
                        rows, a.variant, base, split, metric)
        (out / "tests.json").write_text(json.dumps(tests, indent=1))
        log("\npaired tests (Kendall): selected variant vs baseline")
        for k, v in tests.items():
            if k.endswith("kendall"):
                log(f"  {k.split('|')[0]:<9} {k.split('|')[1]:<40} n={v['n']:<3} diff={v['mean_diff']:+.3f} "
                    f"p={v['p']:.2e} r={v['effect_rb']:+.2f}")
    (out / "config.json").write_text(json.dumps({
        "protocol": a.protocol, "variant": a.variant, "hp": hp, "seeds": seeds,
        "variants": VARIANTS, "device": str(device), "torch": torch.__version__,
        "python": platform.python_version(), "git_commit": os.environ.get("QUEST_COMMIT", "unknown"),
        "smoke": a.smoke}, indent=1))
    log(f"finished in {(time.time() - t0) / 60:.1f} min -> {out}")
    if a.smoke:
        log("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
