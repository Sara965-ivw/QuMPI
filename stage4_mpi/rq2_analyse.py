"""Analysis of the RQ2 preliminary runs (rq2_prelim.py output).

Note on the hypothesis. Measured post hoc, rho_max = t_max / t_min - 1 is an
identity of the measured imbalance, so correlating the two proves nothing.
H2 concerns PRE-execution imbalance, so this script correlates rho_max with the
imbalance PREDICTED by the cost proxy before execution, and separately reports
how well the proxy predicts measured task time (proposal, Phase 4).

Outputs in OUTDIR: summary.txt, per_run.csv, rq2_prelim.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

POLICIES = ("fragment", "count", "lpt", "dynamic")
LABEL = {"fragment": "Fragment per rank group", "count": "Equal task count (METIS-like)",
         "lpt": "Cost-balanced static (LPT)", "dynamic": "Dynamic self-scheduling"}


def per_run(row):
    """Wait is measured as (common elapsed time) - (own busy time), where busy
    time is the sum of the rank's own task durations. Using barrier time alone
    hides waiting that happens inside the task loop (the dynamic policy waits
    for the shared counter there), so it is not comparable across policies."""
    pr = row["per_rank"]
    total = np.array([r["sim_time"] + r["barrier_time"] for r in pr])
    busy = np.array([sum(d for _, d in r["task_times"]) for r in pr])
    wait = total - busy
    work = busy > 0
    rho = wait[work] / busy[work]
    pred = np.array(row["pred_load"], float)
    pred_imb = (pred.max() / pred[pred > 0].min()) if row["policy"] != "dynamic" and (pred > 0).any() else np.nan
    tt = [d for r in pr for _, d in r["task_times"]]
    return {"workload": row["workload"], "family": row["family"], "n_qubits": row["n_qubits"],
            "cut": row["cut"], "ranks": row["ranks"], "policy": row["policy"], "repeat": row["repeat"],
            "n_tasks": row["n_tasks"], "idle_ranks": int((~work).sum()),
            "rho_max": float(rho.max()), "inverted_ranks": int((rho > 1).sum()),
            "efficiency": float(busy.sum() / (len(busy) * total.max())),
            "makespan": float(total.max()), "meas_imbalance": float(busy.max() / busy.mean() - 1),
            "granularity": float(max(tt) / sum(tt)) if tt else np.nan,
            "pred_imbalance": float(pred_imb)}


def boot_median(x, rng, n=10_000):
    x = np.asarray(x, float)
    m = np.median(x[rng.integers(0, len(x), (n, len(x)))], 1)
    return float(np.median(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in a.runs.read_text().splitlines() if l.strip()]
    R = [per_run(r) for r in rows]
    with open(a.out / "per_run.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(R[0])); wr.writeheader(); wr.writerows(R)

    # median over repeats per (workload, ranks, policy)
    cell = {}
    for r in R:
        cell.setdefault((r["workload"], r["ranks"], r["policy"]), []).append(r)
    agg = {k: {m: float(np.median([x[m] for x in v])) for m in
               ("rho_max", "efficiency", "makespan", "pred_imbalance", "idle_ranks")}
           for k, v in cell.items()}
    rng = np.random.default_rng(2026)
    L = [f"RQ2 preliminary: {len(rows)} timed runs, "
         f"{len({r['workload'] for r in R})} workloads, ranks {sorted({r['ranks'] for r in R})}",
         "rho_max = max over working ranks of wait / busy (wait = elapsed - own task time); inversion = rho_max > 1",
         "Medians over workloads [95% bootstrap CI]; each workload = median of its repeats\n"]
    for n in sorted({r["ranks"] for r in R}):
        L.append(f"## {n} ranks")
        L.append(f"{'policy':<32s}{'rho_max':>24s}{'inverted':>10s}{'idle>0':>8s}{'efficiency':>24s}")
        for p in POLICIES:
            ks = [k for k in agg if k[1] == n and k[2] == p]
            if not ks:
                continue
            rm = [agg[k]["rho_max"] for k in ks]; ef = [agg[k]["efficiency"] for k in ks]
            m, lo, hi = boot_median(rm, rng); e, elo, ehi = boot_median(ef, rng)
            L.append(f"{LABEL[p]:<32s}{m:8.2f} [{lo:5.2f},{hi:6.2f}]"
                     f"{np.mean(np.array(rm) > 1):10.0%}{np.mean([agg[k]['idle_ranks'] > 0 for k in ks]):8.0%}"
                     f"{e:9.2f} [{elo:4.2f},{ehi:5.2f}]")
        # paired contrasts over workloads (Wilcoxon), Holm within this rank count
        tests, ps = [], []
        for a_, b_ in (("lpt", "fragment"), ("lpt", "count"), ("dynamic", "lpt")):
            wl = sorted({k[0] for k in agg if k[1] == n})
            xa = [agg[(w, n, a_)]["makespan"] for w in wl if (w, n, a_) in agg and (w, n, b_) in agg]
            xb = [agg[(w, n, b_)]["makespan"] for w in wl if (w, n, a_) in agg and (w, n, b_) in agg]
            d = np.array(xa) - np.array(xb)
            if len(d) >= 5 and np.any(d != 0):
                p = float(stats.wilcoxon(xa, xb).pvalue)
                rpos = stats.rankdata(np.abs(d))
                rb = float((rpos[d > 0].sum() - rpos[d < 0].sum()) / rpos.sum())
                tests.append((a_, b_, float(np.median(np.array(xa) / np.array(xb))), rb)); ps.append(p)
        for (a_, b_, ratio, rb), p, ph in zip(tests, ps, holm(np.array(ps)) if ps else []):
            L.append(f"  makespan {a_} / {b_}: median ratio {ratio:.3f}, Wilcoxon p={p:.2g} "
                     f"(Holm {ph:.2g}), rank-biserial {rb:+.2f}")
        L.append("")
    # H2 mechanism: predicted (pre-execution) imbalance vs rho_max, static policies only
    st_ = [r for r in R if r["policy"] != "dynamic" and np.isfinite(r["pred_imbalance"])]
    if len(st_) > 5:
        rs = stats.spearmanr([r["pred_imbalance"] for r in st_], [r["rho_max"] for r in st_])
        L.append(f"H2 mechanism: Spearman(predicted imbalance, rho_max) = {rs.statistic:.2f} "
                 f"(p={rs.pvalue:.2g}, n={len(st_)} static runs)")
    # proxy validity: predicted task cost vs measured task time
    rhos = []
    for row in rows:
        cost = np.array(row["task_cost"])
        meas = {}
        for pr in row["per_rank"]:
            for i, dt in pr["task_times"]:
                meas[i] = dt
        if len(meas) >= 4:
            idx = sorted(meas)
            rhos.append(stats.spearmanr(cost[idx], [meas[i] for i in idx]).statistic)
    if rhos:
        L.append(f"Cost proxy validity: within-run Spearman(predicted cost, measured task time) "
                 f"median {np.nanmedian(rhos):.2f} (IQR {np.nanpercentile(rhos, 25):.2f} to "
                 f"{np.nanpercentile(rhos, 75):.2f})")
    text = "\n".join(L)
    (a.out / "summary.txt").write_text(text + "\n")
    print(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
        col = {"fragment": "#8a8a85", "count": "#c9a227", "lpt": "#2a78d6", "dynamic": "#1b9e77"}
        ranks = sorted({r["ranks"] for r in R})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=200)
        for j, n in enumerate(ranks):
            for i, p in enumerate(POLICIES):
                ks = [k for k in agg if k[1] == n and k[2] == p]
                if not ks:
                    continue
                m, lo, hi = boot_median([agg[k]["rho_max"] for k in ks], rng)
                x = j * 5 + i
                a1.plot([x, x], [lo, hi], color=col[p], lw=2)
                a1.plot(x, m, "o", color=col[p], ms=7, mec="white", label=LABEL[p] if j == 0 else None)
        a1.axhline(1, color="#333333", ls="--", lw=0.8)
        a1.text(-0.4, 1.04, "inversion threshold ρ = 1", fontsize=7.5)
        a1.set_xticks([j * 5 + 1.5 for j in range(len(ranks))]); a1.set_xticklabels([f"{n} ranks" for n in ranks])
        a1.set_ylabel("Worst-rank wait/work ratio ρ (median, 95% CI)"); a1.set_yscale("log")
        a1.legend(frameon=False, fontsize=7.5)
        for p in ("fragment", "count", "lpt"):
            pts = [(r["pred_imbalance"], r["rho_max"]) for r in st_ if r["policy"] == p]
            if pts:
                a2.scatter(*zip(*pts), s=10, color=col[p], alpha=0.7, label=LABEL[p])
        a2.axhline(1, color="#333333", ls="--", lw=0.8)
        a2.set_xscale("log"); a2.set_yscale("log")
        a2.set_xlabel("Predicted load ratio max/min (before execution)"); a2.set_ylabel("Measured ρ_max")
        for ax in (a1, a2):
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(a.out / "rq2_prelim.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
