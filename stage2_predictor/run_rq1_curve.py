"""RQ1 learning curve: would more training circuits help the Graph Transformer?

Why this run exists (21 Oct 2026). The Delta follow-up (PREREG_delta.md) was not
adopted: gt_delta-pair was not significantly better than layout-ESP on sel_12_14.
Labelling more training circuits costs about 0.7 CPU-hours per circuit (150
CPU-hours for the current 210), so before generating data we measure whether
performance is still rising with training-set size.

Design (training data only; holdout3 is never read; exploratory)
  Train circuits (6-10 q) are subsampled to 25, 50, 75 and 100 per cent,
  stratified by (family, width). For each seed the subsets are NESTED
  (25 % inside 50 % inside 75 % inside 100 %), so a difference between sizes is
  not a difference between unrelated samples. Same validation split for early
  stopping, judged on sel_12_14 (12-14 q), 5 seeds.
  Models: gt_size-pair and gt_delta-pair (the two best in rq1-delta), GBM-layout
  (to see whether trees saturate earlier than the transformer). ESP is constant.

Reading rule (stated before the run)
  Generating more data is worth it for a model if its gap-weighted tau at 100 %
  exceeds that at 50 % (one doubling) by at least 0.02, with the paired bootstrap
  95 % CI over circuits excluding zero. The per-doubling gain also gives a rough
  extrapolation to 2x and 3x data, reported as an extrapolation, not a result.
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
from run_rq1 import HP, score_rows, trainable, write_csv      # noqa: E402
from run_rq1_delta import train                               # noqa: E402
from run_rq1_v2 import predict, standardise_hand              # noqa: E402

FRACS = [0.25, 0.5, 0.75, 1.0]
NEURAL = ["gt_size-pair", "gt_delta-pair"]


def nested_subsets(graphs, seed):
    """Per (family, width) stratum: shuffle once, then take the first ceil(f * n)."""
    rng = np.random.default_rng(10_000 + seed)
    strata = {}
    for g in graphs:
        strata.setdefault((g.family, g.n_qubits), []).append(g)
    order = {k: [v[i] for i in rng.permutation(len(v))] for k, v in strata.items()}
    return {f: [g for v in order.values() for g in v[:max(1, int(np.ceil(f * len(v))))]] for f in FRACS}


def per_circuit(rows, model, split="sel_12_14"):
    acc = {}
    for r in rows:
        if r["model"] == model and r["split"] == split and r.get("kendall_w") is not None:
            acc.setdefault(r["cid"], []).append(r["kendall_w"])
    return {c: float(np.mean(v)) for c, v in acc.items()}


def boot_diff(a, b, n_boot=10_000, seed=2026):
    common = sorted(set(a) & set(b))
    d = np.array([a[c] - b[c] for c in common])
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--smoke", action="store_true")
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
    log(f"=== RQ1 learning curve {time.strftime('%Y-%m-%d %H:%M')} device={device} smoke={a.smoke}")
    pool = load_split(Path(a.train), "train", aggregation="layout")
    small = [g for g in pool if g.n_qubits <= 10]
    large = [g for g in pool if g.n_qubits >= 12]
    train_set, val = stratified_split(small)
    for g in large:
        g.split = "sel_12_14"
    standardise_hand(train_set, val + large)
    log(f"train pool {len(train_set)} val {len(val)} sel_12_14 {len(large)}")

    rows = score_rows("esp_layout", 0, large, B.score_esp)
    sizes = {}
    for seed in seeds:
        subs = nested_subsets(train_set, seed)
        for f in FRACS:
            sub = subs[f]
            sizes[f] = len(sub)
            fit = [g for g in sub if trainable(g)]
            rows += score_rows(f"gbm-layout@{f}", seed, large, B.GBMRanker(seed).fit(fit).score)
            for name in NEURAL:
                m = train(name, sub, val, seed, device, hp, log)
                rows += score_rows(f"{name}@{f}", seed, large, lambda g: predict(m, g, device))
        log(f"  seed {seed} done ({(time.time() - t0) / 60:.1f} min)")
    write_csv(out / "per_circuit.csv", rows)

    esp = per_circuit(rows, "esp_layout")
    log(f"\nsel_12_14 gap-weighted tau by training size (ESP constant at {np.mean(list(esp.values())):.3f})")
    log(f"{'model':<15}" + "".join(f"{f'{int(f * 100)}% (n={sizes[f]})':>18}" for f in FRACS))
    curve, verdict = {}, {}
    for name in NEURAL + ["gbm-layout"]:
        vals = {f: per_circuit(rows, f"{name}@{f}") for f in FRACS}
        curve[name] = {f: float(np.mean(list(v.values()))) for f, v in vals.items()}
        log(f"{name:<15}" + "".join(f"{curve[name][f]:>18.3f}" for f in FRACS))
        gain, lo, hi = boot_diff(vals[1.0], vals[0.5])
        vs_esp = boot_diff(vals[1.0], esp)
        worth = gain >= 0.02 and lo > 0
        verdict[name] = {"gain_per_doubling": gain, "ci": [lo, hi], "worth_more_data": worth,
                         "full_vs_esp": vs_esp,
                         "extrapolated_2x": curve[name][1.0] + gain,
                         "extrapolated_3x": curve[name][1.0] + gain * float(np.log2(3))}
        log(f"    one doubling (50% -> 100%): {gain:+.3f} [{lo:+.3f}, {hi:+.3f}]  worth more data: {worth}")
        log(f"    100% vs ESP: {vs_esp[0]:+.3f} [{vs_esp[1]:+.3f}, {vs_esp[2]:+.3f}];  "
            f"extrapolation (not a result): 2x data {verdict[name]['extrapolated_2x']:.3f}, "
            f"3x data {verdict[name]['extrapolated_3x']:.3f}")
    (out / "curve.json").write_text(json.dumps({"sizes": sizes, "curve": curve, "verdict": verdict,
                                                "esp": float(np.mean(list(esp.values())))}, indent=1))
    (out / "config.json").write_text(json.dumps({
        "hp": hp, "seeds": seeds, "fracs": FRACS, "models": NEURAL + ["gbm-layout"],
        "device": str(device), "torch": torch.__version__, "python": platform.python_version(),
        "git_commit": os.environ.get("QUEST_COMMIT", "unknown"), "smoke": a.smoke}, indent=1))
    log(f"finished in {(time.time() - t0) / 60:.1f} min -> {out}")
    if a.smoke:
        log("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
