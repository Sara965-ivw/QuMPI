"""Card B: minimum versus mean calibration aggregation (proposal lines 446-449 and 561).

Proposal: "a two-qubit gate requires the relaxation time, coherence time and error
rate of two physical qubits to be aggregated into a single node feature, both
minimum and mean aggregations are stored as separate columns so that the choice can
be treated as an ablation rather than as a fixed assumption."

Decisions (22 Oct 2026): B1 (a) the frozen primary recipe; B2 (a) training data only
(train 6-10 q, early stopping on 6-10 q validation, judged on 12-14 q, exploratory);
B3 (b) three conditions, minimum, mean and both as separate columns. The frozen
model's own rule (readout mean, T1 and T2 minimum, "mixed") is run as the reference,
so the three conditions can be read against the model that was tested on holdout3.

Independent variable: aggregation of fragment-layout calibration over the qubits of
a two-qubit gate (readout error, T1, T2); the coupler gate error is a single value.
Dependent variable: gap-weighted Kendall tau on 12-14 q (primary), plain tau, top-1 regret.
Controlled: architecture gt_size, ListNet loss, early stopping on plain tau, all
hyperparameters, data split, 5 seeds (identical to run_rq1_hw for gt_size-layout).
Tests (stated before the run): paired Wilcoxon over circuits (seed-averaged) for
min vs mean (the proposal's question), both vs min and both vs mean, Holm over these
three; each condition vs the frozen rule is reported as a reference without inference.
holdout3 is not read.
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
from quest_phase2.hardware import AGG_CHANNELS                             # noqa: E402
from quest_phase2.data import N_CAND_BASE                                  # noqa: E402
from quest_phase2.metrics import circuit_metrics, paired_test              # noqa: E402
from quest_phase2.models import VARIANTS, GraphTransformer, listnet_loss, to_batch  # noqa: E402
from run_rq1 import HP, score_rows, set_seed, trainable                    # noqa: E402
from run_rq1_hw import holm                                                # noqa: E402
from run_rq1_v2 import predict, report, standardise_hand                   # noqa: E402

CONDITIONS = {"mixed": "layout", "min": "layout_min", "mean": "layout_mean", "both": "layout_both"}
TESTS = [("min", "mean"), ("both", "min"), ("both", "mean")]


def mean_kendall(model, graphs, device):
    v = [circuit_metrics(g.y, predict(model, g, device)).get("kendall") for g in graphs if trainable(g)]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float("-inf")


def train_frozen_recipe(n_cand, train, val, seed, device, hp, log, tag):
    """run_rq1_v2.train_variant('gt_size', ...) with the input width as a parameter."""
    set_seed(seed)
    model = GraphTransformer(hp["d"], hp["heads"], hp["layers"], dropout=hp["dropout"],
                             n_cand=n_cand, **VARIANTS["gt_size"]).to(device)
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
        vk = mean_kendall(model, val, device)
        if vk > best + 1e-4:
            best, bad = vk, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    log(f"    {tag} seed {seed}: {ep + 1} epochs, best early-stop Kendall {best:.3f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reference", default=None, help="runs/rq1_hw/per_circuit.csv (reproducibility check)")
    a = ap.parse_args()
    assert "holdout3" not in str(a.train), "holdout3 is not used by this run"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hp, seeds = dict(HP), [int(s) for s in a.seeds.split(",")]
    if a.smoke:
        hp.update(epochs=2, patience=2); seeds = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(out / "log.txt", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    t0 = time.time()
    log(f"=== Card B aggregation ablation {time.strftime('%Y-%m-%d %H:%M')} device={device} smoke={a.smoke}")
    rows = []
    for cond, agg in CONDITIONS.items():
        pool = load_split(Path(a.train), "train", aggregation=agg)
        small = [g for g in pool if g.n_qubits <= 10]
        large = [g for g in pool if g.n_qubits >= 12]
        train, val = stratified_split(small)
        for g in large:
            g.split = "sel_12_14"
        standardise_hand(train, val + large)
        n_cand = N_CAND_BASE + AGG_CHANNELS[{"layout": "mixed"}.get(agg, agg.replace("layout_", ""))]
        assert train[0].x_cand.shape[-1] == n_cand, (cond, train[0].x_cand.shape, n_cand)
        log(f"[{cond}] aggregation={agg} input channels={n_cand}  train {len(train)} val {len(val)} sel_12_14 {len(large)}")
        for seed in seeds:
            m = train_frozen_recipe(n_cand, train, val, seed, device, hp, log, f"gt_size-{cond}")
            rows += score_rows(f"gt_size-{cond}", seed, val + large, lambda g: predict(m, g, device))
    summ = report(rows, out, log, ("val", "sel_12_14"))
    log("\ncondition        split        gap-weighted tau [95% CI]")
    for r in summ:
        if r["split"] in ("val", "sel_12_14"):
            log(f"{r['model']:<16} {r['split']:<12} {r['kendall_w']:+.3f} [{r['kendall_w_lo']:+.3f},{r['kendall_w_hi']:+.3f}]")
    res = [paired_test(rows, f"gt_size-{x}", f"gt_size-{y}", "sel_12_14", "kendall_w") for x, y in TESTS]
    adj = holm(np.array([r["p"] if np.isfinite(r["p"]) else 1.0 for r in res]))
    tests = {f"{x}_vs_{y}": {**r, "p_holm": float(p)} for (x, y), r, p in zip(TESTS, res, adj)}
    tests["reference_vs_frozen_rule"] = {c: paired_test(rows, f"gt_size-{c}", "gt_size-mixed", "sel_12_14",
                                                        "kendall_w") for c in ("min", "mean", "both")}
    (out / "tests.json").write_text(json.dumps(tests, indent=1))
    log("\npaired tests on sel_12_14 (gap-weighted tau, Holm over the three)")
    for (x, y), r, p in zip(TESTS, res, adj):
        log(f"  {x} vs {y}: n={r['n']} diff={r['mean_diff']:+.3f} p={r['p']:.2e} Holm={p:.2e} r={r['effect_rb']:+.2f}")
    log("reference (no inference): each condition vs the frozen 'mixed' rule")
    for c, r in tests["reference_vs_frozen_rule"].items():
        log(f"  {c} vs mixed: diff={r['mean_diff']:+.3f} p={r['p']:.2e}")
    if a.reference and Path(a.reference).exists():     # 'mixed' repeats gt_size-layout of run_rq1_hw
        import csv
        ref = {}
        with open(a.reference) as fh:
            for r in csv.DictReader(fh):
                if r["model"] == "gt_size-layout" and r["split"] == "sel_12_14" and r["kendall_w"]:
                    ref[(r["cid"], int(r["seed"]))] = float(r["kendall_w"])
        mine = {(r["cid"], r["seed"]): r["kendall_w"] for r in rows
                if r["model"] == "gt_size-mixed" and r["split"] == "sel_12_14" and r.get("kendall_w") is not None}
        common = sorted(set(ref) & set(mine))
        if common:
            d = np.array([abs(mine[k] - ref[k]) for k in common])
            log(f"reproducibility vs run_rq1_hw gt_size-layout: {len(common)} circuit-seeds, "
                f"max |diff| {d.max():.2e}, identical {np.mean(d < 1e-9):.0%}")
    (out / "config.json").write_text(json.dumps({
        "hp": hp, "seeds": seeds, "conditions": CONDITIONS, "tests": TESTS, "device": str(device),
        "torch": torch.__version__, "python": platform.python_version(),
        "git_commit": os.environ.get("QUEST_COMMIT", "unknown"), "smoke": a.smoke}, indent=1))
    log(f"finished in {(time.time() - t0) / 60:.1f} min -> {out}")
    if a.smoke:
        log("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
