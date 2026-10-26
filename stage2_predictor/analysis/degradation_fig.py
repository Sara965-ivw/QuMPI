"""Item 8: degradation of ranking quality (and, when available, MAE) with circuit width
(proposal lines 564-565: "the degradation of both as a function of qubit count").

Reads existing result files only; no model is trained and no label is computed.
  panel (a)  holdout3 (16, 18, 20 q), the frozen primary model and the pre-registered
             baselines, gap-weighted tau per width with 95% bootstrap CIs over circuits.
             The pooled test was pre-registered; the per-width breakdown is exploratory
             (PREREG_holdout3.md, Section 5).
  panel (b)  training-selection run rq1_hw (models trained on 6-10 q): validation 6-10 q
             (in distribution) and 12-14 q (extrapolation), same models and baselines.
  panel (c)  MAE of predicted fidelity per width from run_rq1_mae (Card C part 1), if present.
Models in (a) and (b) are trained on different widths (6-14 q and 6-10 q); the panels
are therefore shown side by side and never joined into one curve.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT, SEED = 10_000, 2026
SHOW = [("gt_size-layout", "Graph Transformer (layout)"), ("gbm-layout", "GBM (layout)"),
        ("esp_layout", "layout-ESP"), ("metis", "METIS"), ("min_overhead", "min overhead"),
        ("random", "random")]


def load(path, metric="kendall_w"):
    """per (model, split, width): per-circuit values averaged over seeds."""
    acc = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get(metric) not in (None, ""):
                acc[(r["model"], r["split"], int(r["n_qubits"]))][r["cid"]].append(float(r[metric]))
    return {k: np.array([np.mean(v) for v in d.values()]) for k, d in acc.items()}


def ci(v, rng):
    m = rng.choice(v, size=(N_BOOT, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h3", required=True, help="runs/rq1_h3/per_circuit.csv")
    ap.add_argument("--hw", required=True, help="runs/rq1_hw/per_circuit.csv")
    ap.add_argument("--mae", default=None, help="runs/rq1_mae/per_circuit.csv (optional)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    h3, hw = load(a.h3), load(a.hw)
    table = []
    panels = [("(a) holdout3, frozen model (trained 6-14 q)", h3, ["holdout3"]),
              ("(b) selection run (trained 6-10 q)", hw, ["val", "sel_12_14"])]
    mae = load(a.mae, "mae") if a.mae and Path(a.mae).exists() else None
    fig, axes = plt.subplots(1, 3 if mae else 2, figsize=(15 if mae else 10.5, 4.2))
    for ax, (title, data, splits) in zip(axes, panels):
        for model, label in SHOW:
            pts = sorted((w, data[(model, s, w)]) for (m, s, w) in data if m == model and s in splits)
            if not pts:
                continue
            ws, stats = [p[0] for p in pts], [ci(p[1], rng) for p in pts]
            for w, (mu, lo, hi), (_, v) in zip(ws, stats, pts):
                table.append({"panel": title, "model": model, "n_qubits": w, "n_circuits": len(v),
                              "kendall_w": mu, "lo": lo, "hi": hi})
            mu = [s[0] for s in stats]
            ax.errorbar(ws, mu, yerr=[[m - s[1] for m, s in zip(mu, stats)], [s[2] - m for m, s in zip(mu, stats)]],
                        marker="o", capsize=3, label=label)
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("circuit width (qubits)")
        ax.set_ylabel("gap-weighted Kendall tau")
    axes[0].legend(fontsize=8, loc="lower left")
    if mae:
        ax = axes[2]
        for model in sorted({m for (m, s, w) in mae}):
            pts = sorted((w, np.concatenate([mae[k] for k in mae if k[0] == model and k[2] == w]))
                         for w in sorted({w for (m, s, w) in mae if m == model}))
            stats = [ci(v, rng) for _, v in pts]
            ws, mu = [p[0] for p in pts], [s[0] for s in stats]
            for w, s, (_, v) in zip(ws, stats, pts):
                table.append({"panel": "(c) MAE", "model": model, "n_qubits": w, "n_circuits": len(v),
                              "mae": s[0], "lo": s[1], "hi": s[2]})
            ax.errorbar(ws, mu, yerr=[[m - s[1] for m, s in zip(mu, stats)], [s[2] - m for m, s in zip(mu, stats)]],
                        marker="o", capsize=3, label=model)
        ax.set_title("(c) MAE of predicted fidelity (Card C part 1)", fontsize=10)
        ax.set_xlabel("circuit width (qubits)"); ax.set_ylabel("MAE"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "degradation.png", dpi=200); fig.savefig(out / "degradation.pdf")
    keys = sorted({k for r in table for k in r})
    with open(out / "degradation_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(table)
    (out / "README.json").write_text(json.dumps({"sources": {"h3": a.h3, "hw": a.hw, "mae": a.mae},
                                                 "bootstrap": [N_BOOT, SEED]}, indent=1))
    print(f"figure and table written to {out}")
    print("DEGRADATION FIGURE FINISHED")


if __name__ == "__main__":
    main()
