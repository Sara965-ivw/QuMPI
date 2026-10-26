"""RQ1 experiment: does learned cut ranking generalise beyond the training widths?

Protocol (fixed before any result is seen)
  * training circuits 6-14 qubits, split by CIRCUIT into train/validation
    (80/20, stratified by family and width, split seed 2026);
  * model selection (early stopping) on validation Kendall tau only;
  * the 16-20 qubit holdout is touched once, after training, for evaluation;
  * five model seeds (0-4) for every learned model; metrics averaged over
    circuits, 95 per cent bootstrap confidence intervals over circuits;
  * paired Wilcoxon signed-rank tests (circuits paired, seeds averaged) of the
    Graph Transformer against every baseline, with rank-biserial effect size;
  * selection-bias control: validation circuits are also scored on their
    g <= 1 candidates only, the regime that most holdout circuits were
    labelled in.

Usage
  python run_rq1.py --train DIR/train --holdout DIR/holdout --out OUT [--smoke]
"""
import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quest_phase2 import baselines as B          # noqa: E402
from quest_phase2.data import load_split, stratified_split   # noqa: E402
from quest_phase2.metrics import METRICS, circuit_metrics, paired_test, summarise  # noqa: E402
from quest_phase2.models import GIN, GraphTransformer, listnet_loss, to_batch  # noqa: E402

HP = dict(d=64, heads=4, layers=3, dropout=0.1, lr=3e-4, weight_decay=1e-4,
          epochs=80, patience=15, batch_circuits=8, tau=0.5)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def trainable(g):
    return g.n_cand >= 2 and float(np.ptp(g.y)) > 0


@torch.no_grad()
def predict(model, g, device):
    model.eval()
    return model(to_batch(g, device)).float().cpu().numpy()


def mean_kendall(model, graphs, device):
    vals = []
    for g in graphs:
        if trainable(g):
            m = circuit_metrics(g.y, predict(model, g, device))
            if "kendall" in m:
                vals.append(m["kendall"])
    return float(np.mean(vals)) if vals else float("-inf")


def train_neural(kind, train, val, seed, device, hp, log):
    set_seed(seed)
    model = (GraphTransformer(hp["d"], hp["heads"], hp["layers"], dropout=hp["dropout"])
             if kind == "graph_transformer" else
             GIN(hp["d"], hp["layers"], dropout=hp["dropout"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    data = [g for g in train if trainable(g)]
    best, best_state, bad, curve = float("-inf"), None, 0, []
    rng = np.random.default_rng(seed)
    for ep in range(hp["epochs"]):
        model.train()
        order = rng.permutation(len(data))
        total, opt_steps = 0.0, 0
        opt.zero_grad()
        for k, i in enumerate(order, 1):
            b = to_batch(data[i], device)
            loss = listnet_loss(model(b), b["y_z"], hp["tau"]) / hp["batch_circuits"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss on {data[i].cid}")
            loss.backward()
            total += float(loss) * hp["batch_circuits"]
            if k % hp["batch_circuits"] == 0 or k == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(); opt_steps += 1
        vk = mean_kendall(model, val, device)
        curve.append({"epoch": ep, "train_loss": total / max(len(data), 1), "val_kendall": vk})
        if vk > best + 1e-4:
            best, bad = vk, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= hp["patience"]:
            break
    model.load_state_dict(best_state)
    log(f"    {kind} seed {seed}: {len(curve)} epochs, best val Kendall {best:.3f}")
    return model, curve


def score_rows(name, seed, graphs, scorer):
    rows = []
    for g in graphs:
        s = np.asarray(scorer(g), float)
        for split, idx in ((g.split, np.arange(g.n_cand)),
                           (g.split + "_g1", np.where(g.cut_g <= 1)[0]) if g.split == "val" else (None, None)):
            if split is None or len(idx) < 2:
                continue
            m = circuit_metrics(g.y[idx], s[idx])
            m.update(model=name, seed=seed, split=split, cid=g.cid, family=g.family,
                     n_qubits=g.n_qubits)
            rows.append(m)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--aggregation", default="both", choices=["both", "min", "mean", "none"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hp = dict(HP)
    seeds = [int(s) for s in a.seeds.split(",")]
    lim_t = lim_h = None
    if a.smoke:
        hp.update(epochs=2, patience=2)
        seeds, lim_t, lim_h = [0], 12, 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(out / "log.txt", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    t0 = time.time()
    log(f"=== RQ1 run {time.strftime('%Y-%m-%d %H:%M')}  device={device}  smoke={a.smoke}  "
        f"aggregation={a.aggregation}")
    pool = load_split(Path(a.train), "train", limit=lim_t, aggregation=a.aggregation)
    hold = load_split(Path(a.holdout), "holdout", limit=lim_h, aggregation=a.aggregation)
    train, val = stratified_split(pool)
    log(f"circuits: train {len(train)}  val {len(val)}  holdout {len(hold)};  cuts: "
        f"train {sum(g.n_cand for g in train)}  val {sum(g.n_cand for g in val)}  "
        f"holdout {sum(g.n_cand for g in hold)}")
    evals = val + hold
    rows, curves = [], {}

    log("baselines")
    for seed in seeds:
        rows += score_rows("random", seed, evals, lambda g: B.score_random(g, seed))
    rows += score_rows("min_overhead", 0, evals, B.score_min_overhead)
    rows += score_rows("metis", 0, evals, B.score_metis)
    for seed in seeds:
        gbm = B.GBMRanker(seed).fit([g for g in train if trainable(g)])
        rows += score_rows("gbm", seed, evals, gbm.score)

    for kind in ("gin", "graph_transformer"):
        log(kind)
        for seed in seeds:
            model, curve = train_neural(kind, train, val, seed, device, hp, log)
            curves[f"{kind}_seed{seed}"] = curve
            rows += score_rows(kind, seed, evals, lambda g: predict(model, g, device))
            if not a.smoke:
                torch.save(model.state_dict(), out / f"{kind}_seed{seed}.pt")

    write_csv(out / "per_circuit.csv", rows)
    summ = summarise(rows, ("model", "split"))
    write_csv(out / "summary.csv", summ)
    write_csv(out / "summary_by_width.csv", summarise(rows, ("model", "split", "n_qubits")))
    write_csv(out / "summary_by_family.csv", summarise(rows, ("model", "split", "family")))
    tests = {}
    for split in ("val", "holdout"):
        for base in ("random", "min_overhead", "metis", "gbm", "gin"):
            for metric in ("kendall", "top1_regret"):
                tests[f"{split}|graph_transformer_vs_{base}|{metric}"] = paired_test(
                    rows, "graph_transformer", base, split, metric)
    (out / "tests.json").write_text(json.dumps(tests, indent=1))
    (out / "curves.json").write_text(json.dumps(curves))
    (out / "config.json").write_text(json.dumps({
        "hp": hp, "seeds": seeds, "split_seed": 2026, "aggregation": a.aggregation,
        "metis_method": B.metis_method_used(pool), "device": str(device),
        "torch": torch.__version__, "python": platform.python_version(),
        "git_commit": os.environ.get("QUEST_COMMIT", "unknown"), "smoke": a.smoke}, indent=1))

    log("\nmodel               split        circuits  Kendall [95% CI]         top-1 regret     top-1 hit")
    for r in summ:
        log(f"{r['model']:<19} {r['split']:<12} {r['n_circuits']:>7}  "
            f"{r['kendall']:+.3f} [{r['kendall_lo']:+.3f},{r['kendall_hi']:+.3f}]   "
            f"{r['top1_regret']:.4f}          {r['top1_hit']:.2f}")
    log("\npaired tests, Graph Transformer vs baseline (Kendall; Wilcoxon p, rank-biserial r)")
    for k, v in tests.items():
        if k.endswith("kendall"):
            log(f"  {k.split('|')[0]:<8} {k.split('|')[1]:<38} n={v['n']:<3} "
                f"diff={v['mean_diff']:+.3f}  p={v['p']:.2e}  r={v['effect_rb']:+.2f}")
    log(f"finished in {(time.time() - t0) / 60:.1f} min -> {out}")
    if a.smoke:
        bad = [r for r in rows if any(isinstance(r.get(m), float) and not np.isfinite(r[m])
                                      for m in METRICS)]
        assert not bad, f"non-finite metrics in smoke run: {bad[:2]}"
        log("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
