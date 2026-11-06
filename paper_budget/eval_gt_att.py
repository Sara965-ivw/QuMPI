"""Evaluate the frozen attenuation models on a test set (notes/PREREG_gt_att.md, sections 2-5).

  QUEST_UNBLIND=holdout3 python -m paper_budget.eval_gt_att --test holdout3
  python -m paper_budget.eval_gt_att --val-check --run runs/paper_budget/gt_att/<smoke folder>
Reads the training run (runs/paper_budget/gt_att/main: config.json and the 10 checkpoints) and never
refits anything. Writes eval_<test>.json next to the checkpoints and refuses to overwrite it.

Primary: seed-averaged predictions, true m, regret at 1e8 and 1e9 (H2, H3); Holm over the four tests.
Descriptive: fit metrics (H1), regret against M0 (H4), per-seed spread, per family, minimum-kappa2 rule,
10% pilot estimate of m.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from paper_budget.analysis.deployable_rule import choose, holm, pilot_m
from paper_budget.att_data import _blind_guard, load_circuit
from paper_budget.config import BUDGETS, DATA_V2, RUNS, SPLIT_SEED
from paper_budget.models_att import (DirectGT, ResidualGT, clip_att, predict_m0, predict_m1,
                                     predict_nn)

N_ARR = np.asarray(BUDGETS, float)
PRIMARY = (5, 6)                       # indices of 1e8 and 1e9 in BUDGETS
H2_MARGIN = 0.98                       # M2 regret at most 98% of M1's (at least 2% lower)
H4_MARGIN = 1.01
N_BOOT = 10_000
NETS = {"M2": ResidualGT, "M3": DirectGT}


# ---------------------------------------------------------------- pure functions (tested)
def log_regret(c, a_hat: np.ndarray, m: float | None = None) -> np.ndarray:
    """(7,) log of MSE(chosen cut) / MSE(best candidate) at each budget, true labels."""
    f = np.exp(-np.asarray(a_hat, float))
    m = c.m if m is None else m
    out = np.empty(len(N_ARR))
    for j, N in enumerate(N_ARR):
        true = c.b2 + c.v / N
        out[j] = np.log(true[choose(f, c.g2, m, N)] / max(true.min(), 1e-300))
    return out


def fit_metrics(circuits, preds: list[np.ndarray]) -> dict:
    a = np.concatenate([c.att for c in circuits])
    p = np.concatenate(preds)
    a_w = np.concatenate([c.att - c.att.mean() for c in circuits])
    p_w = np.concatenate([q - q.mean() for q in preds])
    taus = [stats.kendalltau(q, c.att).statistic for c, q in zip(circuits, preds)]
    taus = [t for t in taus if np.isfinite(t)]
    r2 = lambda y, yh: float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"rmse": float(np.sqrt(np.mean((p - a) ** 2))), "mae": float(np.mean(np.abs(p - a))),
            "r2": r2(a, p), "rmse_within": float(np.sqrt(np.mean((p_w - a_w) ** 2))),
            "mae_within": float(np.mean(np.abs(p_w - a_w))),
            "r2_within": float(1 - np.sum((a_w - p_w) ** 2) / np.sum(a_w ** 2)),
            "kendall_tau_b": float(np.mean(taus)) if taus else float("nan")}


def gm(L: np.ndarray) -> np.ndarray:
    """Geometric mean of regret over circuits from (n, 7) log regrets."""
    return np.exp(L.mean(axis=0))


def boot_ratio(L_a: np.ndarray, L_b: np.ndarray, seed: int = SPLIT_SEED) -> np.ndarray:
    """95% bootstrap CI over circuits of gm(regret a) / gm(regret b), per budget -> (2, 7)."""
    rng = np.random.default_rng(seed)
    n = len(L_a)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    r = np.exp((L_a - L_b)[idx].mean(axis=1))                  # (N_BOOT, 7)
    return np.quantile(r, [0.025, 0.975], axis=0)


def wilcoxon_p(d: np.ndarray) -> float:
    return float(stats.wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0


# ---------------------------------------------------------------- evaluation
def load_test(name: str, only: set | None = None):
    _blind_guard(DATA_V2 / name)                               # frozen holdout3 guard
    C, MU = [], []
    for p in sorted((DATA_V2 / name).glob("*.json")):
        if p.name.endswith(".g1_empty.json"):
            continue
        c = load_circuit(p, name)
        if c is not None and c.usable_eval() and (only is None or c.cid in only):
            C.append(c)
            MU.append(np.array([r["obs_mu"] for r in json.loads(p.read_text())["records"]], float))
    return C, MU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=None)
    ap.add_argument("--val-check", action="store_true",
                    help="pipeline check on the run's own validation circuits (no held-out data)")
    ap.add_argument("--run", default=str(RUNS / "gt_att" / "main"))
    a = ap.parse_args()
    run = Path(a.run)
    assert a.val_check != bool(a.test), "give --test <name> or --val-check"
    out = run / ("eval_valcheck.json" if a.val_check else f"eval_{a.test}.json")
    if out.exists():
        raise FileExistsError(f"{out} exists: the evaluation is run once")
    cfg = json.loads((run / "config.json").read_text())
    ab, w, a_min = (cfg["alpha"], cfg["beta"]), np.asarray(cfg["w"]), cfg["a_min"]
    if a.val_check:
        a.test = "valcheck"
        C, MU = load_test(cfg["train_corpus"], set(cfg["val_cids"]))
    else:
        C, MU = load_test(a.test)                              # holdout3 guard applies here
        seen = set(cfg["train_cids"]) | set(cfg["val_cids"])
        assert not seen & {c.cid for c in C}, "test circuits overlap the training split"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"test {a.test}: {len(C)} circuits; run {run.name} (git {cfg['git']['commit'][:8]})")

    raw = {"M0": [predict_m0(c, ab) for c in C], "M1": [predict_m1(c, ab, w) for c in C]}
    per_seed = defaultdict(list)
    for name, cls in NETS.items():
        for s in cfg["seeds"]:
            ck = torch.load(run / f"{name}_seed{s}.pt", map_location=device)
            assert ck["model"] == name
            model = cls().to(device)
            model.load_state_dict(ck["state_dict"])
            per_seed[name].append([predict_nn(model, c, ab, device) for c in C])
        raw[name] = [np.mean([seed_preds[i] for seed_preds in per_seed[name]], axis=0)
                     for i in range(len(C))]
    preds, n_clip = {}, {}
    for k, P in raw.items():
        cl = [clip_att(q, a_min) for q in P]
        preds[k], n_clip[k] = [q for q, _ in cl], int(sum(n for _, n in cl))

    res = {"test": a.test, "n_circuits": len(C), "run": str(run), "config_git": cfg["git"],
           "clipped": n_clip, "fit": {k: fit_metrics(C, P) for k, P in preds.items()}}
    L = {k: np.array([log_regret(c, q) for c, q in zip(C, P)]) for k, P in preds.items()}
    mo = [int(np.lexsort((c.e, c.g2))[0]) for c in C]
    L["min_kappa2"] = np.array([[np.log((c.b2[k] + c.v[k] / N) / max((c.b2 + c.v / N).min(), 1e-300))
                                 for N in N_ARR] for c, k in zip(C, mo)])
    res["regret_gm"] = {k: gm(v).tolist() for k, v in L.items()}
    res["regret_share_gt_1.10"] = {k: np.mean(v > np.log(1.10), axis=0).tolist() for k, v in L.items()}

    tests, ps = [], []
    for ref in ("M1", "M3"):
        for j in PRIMARY:
            d = L[ref][:, j] - L["M2"][:, j]                    # > 0: M2 better
            tests.append({"ref": ref, "budget": BUDGETS[j], "ratio_M2_over_ref": float(np.exp(-d.mean())),
                          "ci95": boot_ratio(L["M2"], L[ref])[:, j].tolist(), "p": wilcoxon_p(d),
                          "n_differ": int(np.sum(np.abs(d) > 1e-12))})
            ps.append(tests[-1]["p"])
    for t, q in zip(tests, holm(ps)):
        t["p_holm"] = float(q)
    res["tests"] = tests
    h2 = [t for t in tests if t["ref"] == "M1"]
    h3 = [t for t in tests if t["ref"] == "M3"]
    r2w = {k: res["fit"][k]["r2_within"] for k in preds}
    res["H1_met"] = r2w["M2"] > r2w["M1"]
    res["H2_met"] = all(t["ratio_M2_over_ref"] <= H2_MARGIN and t["p_holm"] < 0.05 for t in h2)
    res["H3_met"] = all(t["ratio_M2_over_ref"] < 1.0 and t["p_holm"] < 0.05 for t in h3)
    res["H4_met"] = bool(np.all(gm(L["M2"]) / gm(L["M0"]) <= H4_MARGIN))

    # descriptive: per-seed spread, per family, pilot m
    res["seed_regret_gm"] = {k: [gm(np.array([log_regret(c, clip_att(q, a_min)[0]) for c, q in zip(C, sp)])).tolist()
                                 for sp in per_seed[k]] for k in NETS}
    fam = defaultdict(list)
    for i, c in enumerate(C):
        fam[c.g.family].append(i)
    res["per_family_regret_gm"] = {f: {k: gm(L[k][idx]).tolist() for k in ("M0", "M1", "M2", "M3")}
                                   for f, idx in fam.items()}
    pil = {}
    for k in ("M0", "M1", "M2", "M3"):
        rows = []
        for i, (c, q) in enumerate(zip(C, preds[k])):
            f = np.exp(-q)
            row = []
            for j, N in enumerate(N_ARR):
                n_p = max(int(round(0.1 * N)), 2)
                rng = np.random.default_rng([SPLIT_SEED, i, j])
                m_hat = pilot_m(MU[i][mo[i]], c.g2[mo[i]], f[mo[i]], n_p, rng, 20)
                best = max((c.b2 + c.v / N).min(), 1e-300)
                row.append(np.mean([np.log((c.b2[kk] + c.v[kk] / (N - n_p)) / best)
                                    for kk in (choose(f, c.g2, mh, N - n_p) for mh in m_hat)]))
            rows.append(row)
        pil[k] = gm(np.array(rows)).tolist()
    res["pilot_regret_gm"] = pil

    out.write_text(json.dumps(res, indent=1))
    print("\nfit (seed-averaged, clipped):")
    for k, v in res["fit"].items():
        print(f"  {k}: " + "  ".join(f"{m} {x:.4f}" for m, x in v.items()) + f"  clipped {n_clip[k]}")
    print("\ngeometric-mean regret (true m)")
    print(f"{'budget':>8}" + "".join(f"{k:>12}" for k in L))
    for j, N in enumerate(N_ARR):
        print(f"{N:>8.0e}" + "".join(f"{res['regret_gm'][k][j]:>12.4f}" for k in L))
    print("\nprimary tests (ratio = gm regret M2 / reference; < 1 = M2 better)")
    for t in tests:
        print(f"  M2 vs {t['ref']} at {t['budget']:.0e}: ratio {t['ratio_M2_over_ref']:.4f} "
              f"CI [{t['ci95'][0]:.4f}, {t['ci95'][1]:.4f}]  p {t['p']:.2e}  Holm {t['p_holm']:.2e}  "
              f"circuits with different choices {t['n_differ']}/{len(C)}")
    print(f"\nH1 {res['H1_met']}  H2 {res['H2_met']}  H3 {res['H3_met']}  H4 {res['H4_met']}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
