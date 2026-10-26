"""Budget-dependent cut selection: is the best cut a function of the shot budget?

Question (exploratory, descriptive; no model is involved)
---------------------------------------------------------
Existing cutters choose the cut with the smallest sampling overhead gamma^2.
On a noisy device a cut also carries a bias from hardware noise. For an
observable O with |O| <= 1 estimated with N shots, the mean squared error of
the quasi-probability estimator satisfies

    MSE_k(N) <= b_k^2 + gamma_k^2 / N,           b_k = 2 * TVD_k,

because |E_p[O] - E_q[O]| <= 2 TVD(p, q) for any |O| <= 1 (worst-case bias),
and each QPD sample is bounded by gamma_k in magnitude (variance <= gamma_k^2,
Mitarai and Fujii 2021; Piveteau and Sutter 2024). The bound is linear in
x = 1/N, so the budget-optimal cut is the lower envelope of straight lines:
the minimum-overhead cut wins at small N, the minimum-bias cut wins at large
N, and the decision changes at crossover budgets N*.

This script measures, on labels that already exist:
  Q1  how often the highest-fidelity cut is NOT among the minimum-overhead cuts;
  Q2  where the first crossover N* lies (realistic budgets: 1e3 to 1e6 shots);
  Q3  how much larger the error bound of each single-objective policy is than
      that of the budget-optimal choice, as a function of N.

Rigor notes
-----------
* Default splits are train and holdout (both already used in development).
  holdout2 and the planned holdout3 are reserved for the future end-to-end
  evaluation of the Stage-3 policy and are NOT read here.
* The bias term is a worst-case bound (scale 2). --bias-scale 1 gives a
  typical-observable sensitivity analysis; N* scales as 1 / bias_scale^2.
* The minimum-overhead baseline is given its most generous form: among cuts
  tied at the minimum gamma^2 it is credited with the best one.
* Bootstrap 95% CIs over circuits (10 000 resamples, seed 2026).

Usage
  python budget_crossover.py --corpus ~/quest-mpi/data/corpus_v2 --out OUTDIR
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

GRID = np.logspace(1, 9, 801)                 # shot budgets 1e1 .. 1e9
REPORT_N = (1e3, 1e4, 1e5, 1e6)
TOL = 1e-9
FAMILIES = ("clustered_random", "twolocal", "qaoa_sbm")


def load(corpus: Path, splits: list[str]) -> list[dict]:
    circuits = []
    for split in splits:
        for f in sorted((corpus / split).glob("*.json")):
            if f.name.endswith(".g1_empty.json"):
                continue
            d = json.loads(f.read_text())
            recs = d.get("records") or []
            if len(recs) < 2:
                continue
            meta = d.get("circuit_meta") or recs[0].get("circuit_meta", {})
            circuits.append({
                "id": f.stem, "split": split,
                "family": meta.get("family") or next(
                    (fam for fam in FAMILIES if f"_{fam}_" in f.stem), "unknown"),
                "n_qubits": int(meta.get("n_qubits") or re.search(r"_n(\d+)_", f.stem).group(1)),
                "F": np.array([r["pst_hellinger"] for r in recs], float),
                "tvd": np.array([r["tv_distance"] for r in recs], float),
                "g2": np.array([2.0 ** r["log2_sampling_overhead"] for r in recs], float),
                "type": [r["cut_type"] for r in recs],
            })
    return circuits


def bound(b, g2, N):
    """MSE bound for every cut (rows) at every budget (columns)."""
    return (b ** 2)[:, None] + g2[:, None] / np.atleast_1d(N)[None, :]


def analyse(c: dict, scale: float) -> dict:
    F, g2 = c["F"], c["g2"]
    b = scale * c["tvd"]
    minov = np.flatnonzero(g2 <= g2.min() * (1 + 1e-12))
    best_f = int(np.argmax(F))
    # generous minimum-overhead policy: best fidelity within the tied set
    k_mo = int(minov[np.argmax(F[minov])])
    # fidelity-only policy: highest fidelity, ties to lower overhead
    cand = np.flatnonzero(F >= F.max() - TOL)
    k_fo = int(cand[np.argmin(g2[cand])])

    B = bound(b, g2, GRID)                          # (K, len(GRID))
    opt = B.min(0)
    k_opt = B.argmin(0)
    switches = int(np.sum(k_opt[1:] != k_opt[:-1]))
    leaves = np.flatnonzero(~np.isin(k_opt, minov))
    n_star = float(GRID[leaves[0]]) if leaves.size else np.inf

    out = {"id": c["id"], "split": c["split"], "family": c["family"], "n_qubits": c["n_qubits"],
           "n_cand": len(F), "n_minov_tied": len(minov),
           "gamma2_min": float(g2.min()), "gamma2_of_best_F": float(g2[best_f]),
           "F_best": float(F.max()), "F_minov": float(F[k_mo]),
           "F_regret_minov": float(F.max() - F[k_mo]),
           "best_F_in_minov": bool(best_f in set(minov) or F[k_mo] >= F.max() - TOL),
           "n_star_first": n_star, "n_switches": switches,
           "n_distinct_optimal": int(len(np.unique(k_opt)))}
    for N in REPORT_N:
        i = int(np.argmin(np.abs(GRID - N)))
        out[f"ratio_minov_{int(N):d}"] = float(B[k_mo, i] / opt[i])
        out[f"ratio_fidonly_{int(N):d}"] = float(B[k_fo, i] / opt[i])
    out["_curve_minov"] = B[k_mo] / opt
    out["_curve_fidonly"] = B[k_fo] / opt
    return out


def boot_frac(x, rng, n=10_000):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_median(x, rng, n=10_000):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = np.median(x[idx], 1)
    return float(np.median(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(rows, label, rng, lines):
    if not rows:
        return
    ns = np.array([r["n_star_first"] for r in rows])
    fin = np.isfinite(ns)
    lines.append(f"\n## {label}  (circuits: {len(rows)})")
    m, lo, hi = boot_frac([not r["best_F_in_minov"] for r in rows], rng)
    lines.append(f"Q1  best-fidelity cut NOT among min-overhead cuts: {m:6.1%}  [{lo:.1%}, {hi:.1%}]")
    m, lo, hi = boot_frac([r["F_regret_minov"] for r in rows], rng)
    lines.append(f"    fidelity lost by min-overhead (generous): mean {m:.4f}  [{lo:.4f}, {hi:.4f}]")
    for name, sel in (("N* < 1e3", fin & (ns < 1e3)),
                      ("1e3 <= N* <= 1e6", fin & (ns >= 1e3) & (ns <= 1e6)),
                      ("1e6 < N* <= 1e9", fin & (ns > 1e6)),
                      ("never (min-overhead optimal at all budgets)", ~fin)):
        m, lo, hi = boot_frac(sel, rng)
        lines.append(f"Q2  {name:<44s} {m:6.1%}  [{lo:.1%}, {hi:.1%}]")
    if fin.any():
        md, lo, hi = boot_median(ns[fin], rng)
        q1, q3 = np.percentile(ns[fin], [25, 75])
        lines.append(f"    median N* (circuits with a crossover): {md:.3g}  [{lo:.3g}, {hi:.3g}],"
                     f" IQR {q1:.3g} to {q3:.3g}")
    lines.append(f"    distinct optimal cuts across 1e1..1e9 shots: mean "
                 f"{np.mean([r['n_distinct_optimal'] for r in rows]):.2f}")
    for N in REPORT_N:
        k = int(N)
        a = np.array([r[f"ratio_minov_{k}"] for r in rows])
        f = np.array([r[f"ratio_fidonly_{k}"] for r in rows])
        lines.append(f"Q3  N={N:7.0e}: min-overhead bound / optimal  median {np.median(a):6.2f}, "
                     f">=1.1x in {np.mean(a >= 1.1):5.1%} | fidelity-only median {np.median(f):6.2f}, "
                     f">=1.1x in {np.mean(f >= 1.1):5.1%}")


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), dpi=200)
    # Left: share of circuits where a single-objective policy's error bound is
    # at least 10% above the budget-optimal choice (median curves hide the
    # effect because half the circuits have no crossover at all).
    for key, lab, col in (("_curve_minov", "Minimum overhead only (current tools)", "#8a8a85"),
                          ("_curve_fidonly", "Fidelity only", "#2a78d6")):
        M = np.vstack([r[key] for r in rows])
        a1.plot(GRID, 100 * np.mean(M >= 1.1, 0), color=col, lw=2, label=lab)
    a1.axvspan(1e3, 1e6, color="#eeeeeb", zorder=0)
    a1.set_xscale("log"); a1.set_ylim(0, 100)
    a1.set_xlabel("Shot budget N")
    a1.set_ylabel("Circuits with error bound \u2265 1.1\u00d7 optimal (%)")
    a1.legend(frameon=False, fontsize=8, loc="upper center")
    ns = np.array([r["n_star_first"] for r in rows]); ns = ns[np.isfinite(ns)]
    if ns.size:
        a2.hist(np.log10(ns), bins=np.arange(1, 9.25, 0.25), color="#2a78d6", edgecolor="white")
    a2.axvspan(3, 6, color="#eeeeeb", zorder=0)
    a2.set_ylim(0, a2.get_ylim()[1] * 1.15)
    a2.text(4.5, a2.get_ylim()[1] * 0.95, "typical budgets 10\u00b3\u201310\u2076", fontsize=7.5,
            color="#555555", ha="center")
    a2.set_xlabel("log10 first crossover budget N*"); a2.set_ylabel("Circuits")
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--splits", default="train,holdout")
    ap.add_argument("--bias-scale", type=float, default=2.0)
    a = ap.parse_args()
    splits = a.splits.split(",")
    assert "holdout2" not in splits and "holdout3" not in splits, \
        "holdout2/holdout3 are reserved for the end-to-end evaluation"
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    rows = [analyse(c, a.bias_scale) for c in load(a.corpus, splits)]
    keys = [k for k in rows[0] if not k.startswith("_")]
    with open(a.out / "per_circuit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    lines = [f"Budget crossover analysis  splits={splits}  bias_scale={a.bias_scale}  "
             f"(N* scales as 1/bias_scale^2)"]
    summarise(rows, "ALL", rng, lines)
    for s in splits:
        summarise([r for r in rows if r["split"] == s], f"split {s}", rng, lines)
    for fam in sorted({r["family"] for r in rows}):
        summarise([r for r in rows if r["family"] == fam], f"family {fam}", rng, lines)
    text = "\n".join(lines)
    (a.out / "summary.txt").write_text(text + "\n")
    print(text)
    try:
        figure(rows, a.out / "crossover.png")
        print(f"\nwritten: {a.out}/summary.txt, per_circuit.csv, crossover.png")
    except ImportError:
        print(f"\nwritten: {a.out}/summary.txt, per_circuit.csv "
              "(figure skipped: pip install matplotlib)")


if __name__ == "__main__":
    main()
