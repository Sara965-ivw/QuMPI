"""Recompute bootstrap confidence intervals with the circuit as the resampling unit.

Original method (quest_phase2/metrics.py: summarise -> bootstrap_ci): every row of
per_circuit.csv is one (model, seed, split, circuit) value, and the bootstrap
resamples ROWS. A model trained with 5 seeds contributes 5 rows per circuit, a
deterministic baseline 1 row, so the seeded models' intervals treat seeds of the
same circuit as independent observations (pseudo-replication).

Corrected method: within each (model, split), average the metric over seeds for
each circuit first, then resample CIRCUITS with replacement (2 000 resamples,
numpy default_rng(0), percentile 2.5 / 97.5) -- the same resample count, seed and
percentile rule as the original, so only the resampling unit changes. Seed
averaging is also what metrics.paired_test already does before the Wilcoxon test.

Inputs are read only. Nothing in the original run directories is written.
Usage: python recompute_ci.py RUNS_DIR OUT_DIR
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = ["phase2", "rq1_select", "rq1_final_gt_size", "rq1_final_gt_hybrid", "rq1_hw",
        "rq1_h3", "rq1_delta", "rq1_agg", "rq1_curve"]
METRICS = ["kendall", "kendall_w", "spearman", "top1_regret", "top1_hit", "ndcg3"]


def boot(values, n_boot=2000, seed=0, alpha=0.05):
    """Identical to quest_phase2.metrics.bootstrap_ci."""
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], float)
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def num(x):
    return None if x in ("", None) else float(x)


def main(runs_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for run in RUNS:
        pc = runs_dir / run / "per_circuit.csv"
        if not pc.exists():
            report.append(f"{run}: per_circuit.csv missing -> skipped")
            continue
        rows = list(csv.DictReader(open(pc)))
        metrics = [m for m in METRICS if m in rows[0]]
        groups = defaultdict(list)
        for r in rows:
            groups[(r["model"], r["split"])].append(r)
        orig_sum = {}
        sp = runs_dir / run / "summary.csv"
        if sp.exists():
            for r in csv.DictReader(open(sp)):
                orig_sum[(r["model"], r["split"])] = r
        out, max_repro = [], 0.0
        for (model, split), rs in sorted(groups.items()):
            seeds = sorted({r["seed"] for r in rs})
            rec = {"run": run, "model": model, "split": split,
                   "n_circuits": len({r["cid"] for r in rs}), "n_seeds": len(seeds)}
            for m in metrics:
                # original method, recomputed from the CSV (row-level bootstrap)
                o_mean, o_lo, o_hi = boot([num(r[m]) for r in rs])
                # check against the stored summary.csv, if any
                s = orig_sum.get((model, split))
                if s is not None and s.get(m + "_lo") not in (None, "", "nan"):
                    for a, b in ((o_mean, s[m]), (o_lo, s[m + "_lo"]), (o_hi, s[m + "_hi"])):
                        if not math.isnan(a):
                            max_repro = max(max_repro, abs(a - float(b)))
                # corrected method: seed-average per circuit, resample circuits
                per_c = defaultdict(list)
                for r in rs:
                    if num(r[m]) is not None and not math.isnan(num(r[m])):
                        per_c[r["cid"]].append(num(r[m]))
                c_mean, c_lo, c_hi = boot([float(np.mean(v)) for v in per_c.values()])
                rec.update({f"{m}_mean_original": o_mean, f"{m}_lo_original": o_lo,
                            f"{m}_hi_original": o_hi, f"{m}_mean_corrected": c_mean,
                            f"{m}_lo_corrected": c_lo, f"{m}_hi_corrected": c_hi,
                            f"{m}_width_ratio": ((c_hi - c_lo) / (o_hi - o_lo))
                            if (o_hi - o_lo) > 0 else float("nan")})
            out.append(rec)
        keys = list(out[0].keys())
        for r in out:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out_dir / f"{run}_ci_corrected.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(out)
        note = (f"max |recomputed original - summary.csv| = {max_repro:.2e}" if orig_sum
                else "no summary.csv to check against")
        report.append(f"{run}: {len(rows)} rows, {len(out)} model-split groups; {note}")
    (out_dir / "recompute_log.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
