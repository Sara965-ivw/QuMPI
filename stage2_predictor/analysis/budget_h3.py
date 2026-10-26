"""Pre-registered family B on holdout3: is the best cut a function of the shot budget?

Pre-registration: PREREG_holdout3.md, Section 4 (exact MSE) and Section 5 (B1-B3).
No model is involved; every quantity comes from the v3.1 labels.

Exact finite-shot error (Section 4)
-----------------------------------
For a candidate cut k with sampling overhead gamma_k^2 and a Pauli-Z string S,
every QPD sample takes the value +-gamma_k, so for N samples

    MSE_kS(N) = (mu_kS - mu*_S)^2 + (gamma_k^2 - mu_kS^2) / N,

mu_kS = expectation of the noisy unclipped quasi-distribution (obs_mu),
mu*_S = ideal expectation (obs_ideal). The per-circuit error is the basket mean

    MSE_k(N) = b2_k + v_k / N,   b2_k = mean_S (mu_kS - mu*_S)^2,
                                 v_k  = mean_S (gamma_k^2 - mu_kS^2).

In x = 1/N every cut is a straight line (intercept b2_k, slope v_k); the
budget-optimal cut is the lower envelope.

Pre-registered hypotheses (decision: lower 95% bootstrap bound of the share > 20%)
  B1  argmin_k MSE_k(1e3) differs from argmin_k MSE_k(1e6).
  B2  at N = 1e6, minimum-overhead rule has MSE >= 1.1 x optimal.
  B3  at N = 1e3, highest-fidelity rule has MSE >= 1.1 x optimal.

Implementation details not spelled out in Part A, fixed BEFORE this script
first read holdout3 (logged in the PREREG deviations, frozen by tag
prereg-holdout3-Bfam):
  * Ties: the optimal set at N is {k : MSE_k <= min MSE (1 + 1e-9)}; B1 counts a
    change only if the optimal sets at 1e3 and 1e6 are disjoint (conservative).
  * Minimum-overhead rule (as pre-registered): credited with the best cut
    (lowest MSE at that N) among cuts tied at the minimum gamma^2.
  * Highest-fidelity rule: the same generosity, best cut among cuts tied at
    the maximum Hellinger fidelity (tolerance 1e-12).
  * Per-shot variances gamma^2 - mu^2 < 0 can only arise numerically; they are
    clipped at 0 and counted.
  * Bounds: 2.5th / 97.5th percentiles of 10 000 bootstrap resamples over
    circuits, seed 2026. A Wilson interval is reported as a sensitivity check.

Everything else (families, widths, budget curves, crossover budgets,
shot-equivalent cost, Hellinger vs MSE ranking) is EXPLORATORY.

Usage
  QUEST_UNBLIND=holdout3 python budget_h3.py --corpus DATA/corpus_v2 --out OUT
  python budget_h3.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import numpy as np

N_SMALL, N_LARGE = 1e3, 1e6
THRESH_RATIO, FLOOR = 1.1, 0.20
REL_TIE, F_TIE = 1e-9, 1e-12
GRID = np.logspace(1, 9, 801)
REPORT_N = (1e3, 1e4, 1e5, 1e6)
FAMILIES = ("clustered_random", "twolocal", "qaoa_sbm")
N_BOOT, SEED = 10_000, 2026


# ------------------------------------------------------------------ data
def load_circuit(path: Path) -> dict:
    d = json.loads(path.read_text())
    recs = d["records"]
    meta = d.get("circuit_meta") or {}
    mu_star = np.asarray(d["obs_ideal"], float)
    mu = np.asarray([r["obs_mu"] for r in recs], float)
    assert mu.shape == (len(recs), len(mu_star)), f"{path.name}: basket size mismatch"
    return {"id": path.stem,
            "family": meta.get("family") or next((f for f in FAMILIES if f"_{f}_" in path.stem), "unknown"),
            "n_qubits": int(meta.get("n_qubits") or re.search(r"_n(\d+)_", path.stem).group(1)),
            "mu": mu, "mu_star": mu_star,
            "g2": np.asarray([r["gamma2"] for r in recs], float),
            "F": np.asarray([r["pst_hellinger"] for r in recs], float),
            "cut_type": [r["cut_type"] for r in recs]}


# -------------------------------------------------------------- analysis
def terms(c: dict):
    """b2_k (squared bias) and v_k (per-shot variance), both basket means."""
    b2 = np.mean((c["mu"] - c["mu_star"][None, :]) ** 2, axis=1)
    var = c["g2"][:, None] - c["mu"] ** 2
    n_neg = int(np.sum(var < 0))
    v = np.mean(np.clip(var, 0.0, None), axis=1)
    return b2, v, n_neg


def mse(b2, v, N):
    return b2[:, None] + v[:, None] / np.atleast_1d(np.asarray(N, float))[None, :]


def opt_set(m: np.ndarray) -> set:
    return set(np.flatnonzero(m <= m.min() * (1 + REL_TIE)).tolist())


def rule_pick(m: np.ndarray, allowed: np.ndarray) -> int:
    """Generous rule: best (lowest-MSE) cut among the allowed (tied) set."""
    return int(allowed[np.argmin(m[allowed])])


def analyse(c: dict) -> dict:
    b2, v, n_neg = terms(c)
    g2, F = c["g2"], c["F"]
    mo = np.flatnonzero(g2 <= g2.min() * (1 + 1e-12))
    hf = np.flatnonzero(F >= F.max() - F_TIE)
    m_s, m_l = mse(b2, v, N_SMALL)[:, 0], mse(b2, v, N_LARGE)[:, 0]
    S, L = opt_set(m_s), opt_set(m_l)
    k_mo_l, k_hf_s = rule_pick(m_l, mo), rule_pick(m_s, hf)

    M = mse(b2, v, GRID)                                  # (K, grid)
    opt = M.min(0)
    k_opt = M.argmin(0)
    leaves = np.flatnonzero(~np.isin(k_opt, mo))
    out = {"id": c["id"], "family": c["family"], "n_qubits": c["n_qubits"],
           "n_cand": len(g2), "n_basket": len(c["mu_star"]), "n_neg_var": n_neg,
           "n_distinct_overhead": len(np.unique(np.round(np.log2(g2), 9))),
           # pre-registered
           "B1_change": bool(S.isdisjoint(L)),
           "B2_ratio_minov_1e6": float(m_l[k_mo_l] / m_l.min()),
           "B3_ratio_hifid_1e3": float(m_s[k_hf_s] / m_s.min()),
           # exploratory
           "gamma2_opt_1e3": float(g2[min(S)]), "gamma2_opt_1e6": float(g2[min(L)]),
           "first_N_leaving_minov": float(GRID[leaves[0]]) if leaves.size else math.inf,
           "n_distinct_optimal_grid": int(len(np.unique(k_opt))),
           "hifid_is_mse_opt_inf": bool(int(np.argmin(b2)) in set(hf.tolist())),
           "spearman_F_vs_neg_b2": _spearman(F, -b2)}
    for N in REPORT_N:
        i = int(np.argmin(np.abs(GRID - N)))
        mN = M[:, i]
        k_mo, k_hf = rule_pick(mN, mo), rule_pick(mN, hf)
        out[f"ratio_minov_{int(N)}"] = float(mN[k_mo] / opt[i])
        out[f"ratio_hifid_{int(N)}"] = float(mN[k_hf] / opt[i])
        # shot-equivalent cost of the minimum-overhead rule: budget N' at which
        # that cut reaches the optimal error of budget N (inf if its bias floor
        # is above it)
        gap = opt[i] - b2[k_mo]
        out[f"shot_mult_minov_{int(N)}"] = float(v[k_mo] / gap / N) if gap > 0 else math.inf
    out["_curve_minov"] = np.array([M[rule_pick(M[:, j], mo), j] for j in range(len(GRID))]) / opt
    out["_curve_hifid"] = np.array([M[rule_pick(M[:, j], hf), j] for j in range(len(GRID))]) / opt
    return out


def _spearman(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------ statistics
def boot_share(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def decide(name, flags, rng):
    flags = np.asarray(flags, bool)
    m, lo, hi = boot_share(flags, rng)
    wlo, whi = wilson(int(flags.sum()), len(flags))
    return {"hypothesis": name, "n": int(len(flags)), "k": int(flags.sum()),
            "share": m, "boot_lo": lo, "boot_hi": hi, "wilson_lo": wlo, "wilson_hi": whi,
            "floor": FLOOR, "supported": bool(lo > FLOOR)}


def summarise(rows, label, rng, lines):
    if not rows:
        return
    lines.append(f"\n## {label} (circuits: {len(rows)})")
    for key, name in (("B1_change", "B1 optimal cut changes 1e3 -> 1e6"),):
        m, lo, hi = boot_share([r[key] for r in rows], rng)
        lines.append(f"  {name:<44} {m:6.1%} [{lo:.1%}, {hi:.1%}]")
    for N in REPORT_N:
        a = np.array([r[f"ratio_minov_{int(N)}"] for r in rows])
        f = np.array([r[f"ratio_hifid_{int(N)}"] for r in rows])
        lines.append(f"  N={N:7.0e}  min-overhead >=1.1x opt {np.mean(a >= THRESH_RATIO):6.1%} "
                     f"(median ratio {np.median(a):.2f}) | highest-fidelity >=1.1x opt "
                     f"{np.mean(f >= THRESH_RATIO):6.1%} (median {np.median(f):.2f})")
    ns = np.array([r["first_N_leaving_minov"] for r in rows])
    fin = np.isfinite(ns)
    lines.append(f"  optimal cut leaves the min-overhead set within 1e1..1e9 shots: {fin.mean():.1%}"
                 + (f"; median first budget {np.median(ns[fin]):.3g}" if fin.any() else ""))
    sm = np.array([r["shot_mult_minov_1000000"] for r in rows])
    lines.append(f"  min-overhead at N=1e6: shots needed to match the optimal error: "
                 f"median {np.median(sm):.3g}x; unreachable (bias floor) in {np.mean(~np.isfinite(sm)):.1%}")


# ---------------------------------------------------------------- figure
def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), dpi=200)
    for key, lab, col, Npt in (("_curve_minov", "Minimum overhead (current tools)", "#6b6b66", N_LARGE),
                               ("_curve_hifid", "Highest fidelity", "#2a78d6", N_SMALL)):
        share = 100 * np.mean(np.vstack([r[key] for r in rows]) >= THRESH_RATIO, 0)
        a1.plot(GRID, share, color=col, lw=2, label=lab)
        j = int(np.argmin(np.abs(GRID - Npt)))
        a1.plot([GRID[j]], [share[j]], "o", color=col, ms=5)
    a1.axhline(100 * FLOOR, color="#c0392b", lw=1, ls="--")
    a1.text(12, 100 * FLOOR + 2, "pre-registered 20% floor", color="#c0392b", fontsize=7.5)
    a1.axvspan(N_SMALL, N_LARGE, color="#eeeeeb", zorder=0)
    a1.set_xscale("log"); a1.set_ylim(0, 100)
    a1.set_xlabel("Shot budget N")
    a1.set_ylabel("Circuits with MSE ≥ 1.1× optimal (%)")
    a1.legend(frameon=False, fontsize=8, loc="upper center")
    ns = np.array([r["first_N_leaving_minov"] for r in rows]); ns = ns[np.isfinite(ns)]
    if ns.size:
        a2.hist(np.log10(ns), bins=np.arange(1, 9.25, 0.25), color="#2a78d6", edgecolor="white")
    a2.yaxis.set_major_locator(MaxNLocator(integer=True))
    a2.axvspan(3, 6, color="#eeeeeb", zorder=0)
    a2.set_xlabel("log10 budget at which the optimal cut stops being minimum-overhead")
    a2.set_ylabel("Circuits")
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path)


# --------------------------------------------------------------- selftest
def selftest():
    """Two cuts with a known crossover: cut 0 unbiased, gamma^2 = 9; cut 1
    biased (b2 = 1e-3), gamma^2 = 1. Lines cross at N = 8000 (approximately,
    because v_k = gamma^2 - mu^2)."""
    mu_star = np.array([0.5])
    c = {"id": "t_clustered_random_n16_s1", "family": "clustered_random", "n_qubits": 16,
         "mu_star": mu_star, "mu": np.array([[0.5], [0.5 + math.sqrt(1e-3)]]),
         "g2": np.array([9.0, 1.0]), "F": np.array([0.99, 0.90]), "cut_type": ["gate", "wire"]}
    r = analyse(c)
    b2, v, _ = terms(c)
    n_star = (v[0] - v[1]) / (b2[1] - b2[0])
    assert abs(b2[1] - 1e-3) < 1e-12 and 7000 < n_star < 9000, n_star
    assert r["B1_change"], r
    # at 1e6 the min-overhead cut (1) is bias-limited: ratio = (1e-3 + ~1e-6) / (8.75e-6)
    assert r["B2_ratio_minov_1e6"] > 100, r["B2_ratio_minov_1e6"]
    # at 1e3 the highest-fidelity cut (0) pays variance: 8.75e-3 / (1e-3 + 0.7e-3)
    assert 4.5 < r["B3_ratio_hifid_1e3"] < 5.5, r["B3_ratio_hifid_1e3"]
    # MSE formula against Monte Carlo for one observable
    rng = np.random.default_rng(0)
    g, mu = 3.0, 0.4                                          # P(+g) = (1 + mu/g) / 2
    est = (rng.random((20000, 50)) < (1 + mu / g) / 2).astype(float) * 2 * g - g
    mc = np.mean((est.mean(1) - 0.5) ** 2)
    exact = (mu - 0.5) ** 2 + (g * g - mu * mu) / 50
    assert abs(mc - exact) / exact < 0.05, (mc, exact)
    # ties: identical cuts -> optimal sets overlap -> no change counted
    c2 = {**c, "mu": np.array([[0.5], [0.5]]), "g2": np.array([4.0, 4.0]), "F": np.array([0.9, 0.9])}
    assert not analyse(c2)["B1_change"]
    print("SELFTEST PASSED")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    assert os.environ.get("QUEST_UNBLIND") == "holdout3", "family B reads holdout3: needs QUEST_UNBLIND=holdout3"
    a.out.mkdir(parents=True, exist_ok=True)
    tests_path = a.out / "budget_h3_tests.json"
    assert not tests_path.exists(), f"{tests_path} exists: family B is evaluated once"
    sel = json.loads((a.corpus / "holdout3_selection.json").read_text())["circuits"]
    circuits = [load_circuit(a.corpus / "holdout3" / f"{cid}.json") for cid in sel]
    assert len(circuits) == len(sel) == 143, f"{len(circuits)} circuits vs 143 selected"
    rows = [analyse(c) for c in circuits]
    rng = np.random.default_rng(SEED)
    res = {"n_circuits": len(rows),
           "B1": decide("B1: optimal cut changes between N=1e3 and N=1e6", [r["B1_change"] for r in rows], rng),
           "B2": decide("B2: minimum overhead >= 1.1x optimal MSE at N=1e6",
                        [r["B2_ratio_minov_1e6"] >= THRESH_RATIO for r in rows], rng),
           "B3": decide("B3: highest fidelity >= 1.1x optimal MSE at N=1e3",
                        [r["B3_ratio_hifid_1e3"] >= THRESH_RATIO for r in rows], rng),
           "diagnostics": {"negative_variance_terms": int(sum(r["n_neg_var"] for r in rows)),
                           "mean_candidates": float(np.mean([r["n_cand"] for r in rows])),
                           "mean_basket": float(np.mean([r["n_basket"] for r in rows]))}}
    tests_path.write_text(json.dumps(res, indent=1))
    keys = [k for k in rows[0] if not k.startswith("_")]
    with open(a.out / "per_circuit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    lines = ["PRE-REGISTERED FAMILY B (holdout3, exact finite-shot MSE; supported if lower bound > 20%)"]
    for b in ("B1", "B2", "B3"):
        v = res[b]
        lines.append(f"  {v['hypothesis']:<52} {v['k']:3d}/{v['n']} = {v['share']:6.1%} "
                     f"[{v['boot_lo']:.1%}, {v['boot_hi']:.1%}] (Wilson {v['wilson_lo']:.1%}-{v['wilson_hi']:.1%})"
                     f"  -> {'SUPPORTED' if v['supported'] else 'NOT SUPPORTED'}")
    lines.append(f"  diagnostics: {res['diagnostics']}")
    lines.append("\nEXPLORATORY")
    summarise(rows, "all", rng, lines)
    for fam in FAMILIES:
        summarise([r for r in rows if r["family"] == fam], f"family {fam}", rng, lines)
    for n in sorted({r["n_qubits"] for r in rows}):
        summarise([r for r in rows if r["n_qubits"] == n], f"width {n}", rng, lines)
    agree = np.mean([r["hifid_is_mse_opt_inf"] for r in rows])
    rho = np.nanmedian([r["spearman_F_vs_neg_b2"] for r in rows])
    lines.append(f"\nHellinger vs MSE: highest-fidelity cut is also the lowest-bias cut in {agree:.1%} of circuits; "
                 f"median Spearman(F, -bias^2) = {rho:.2f}")
    text = "\n".join(lines)
    (a.out / "summary.txt").write_text(text + "\n")
    print(text)
    try:
        figure(rows, a.out / "budget_h3.png")
        print(f"\nwritten: {a.out}/budget_h3_tests.json, summary.txt, per_circuit.csv, budget_h3.png")
    except ImportError:
        print(f"\nwritten: {a.out}/budget_h3_tests.json, summary.txt, per_circuit.csv (no figure: matplotlib missing)")


if __name__ == "__main__":
    main()
