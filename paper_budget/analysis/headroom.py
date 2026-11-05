"""Headroom analysis (exploratory; holdout3, scheme v3.1, already unblinded).

Q1. Within a circuit, how much of the cut-induced attenuation does layout-ESP explain?
Q2. Is the part that ESP misses predictable from simple cut features (learnable headroom)?
Q3. Choosing a cut by budget with an ESP-based bias estimate: how far from the oracle?
    (ESP is given the true mu* as weights, so Q3 measures headroom in the attenuation only.)
Predictions were committed beforehand in notes/PREDICTIONS_headroom.md.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

from paper_budget.config import ROOT, DATA_V2, BUDGETS, SPLIT_SEED
from paper_budget.targets import bias_variance, gamma2

sys.path.insert(0, str(ROOT / "stage2_predictor"))
from quest_phase2.data import build_graph  # noqa: E402  (frozen thesis code)

BIG = 0.2   # |mu*| above which mu/mu* is well defined (RESULTS_obs_structure.md)
FEATS = ("g", "w", "log2_sampling_overhead", "routing_overhead_2q",
         "n_2q_mapped", "depth_a", "depth_b")


def load(path):
    """One circuit: attenuation per candidate, ESP error mass, cut features, bias/variance."""
    d = json.loads(path.read_text())
    if not d.get("records"):
        return None
    g = build_graph(path, "holdout3", aggregation="layout")
    ideal = np.asarray(d["obs_ideal"], float)
    mu = np.array([r["obs_mu"] for r in d["records"]], float)          # (K, S)
    big = np.abs(ideal) > BIG
    att = (-np.log(np.clip(mu[:, big] / ideal[big], 1e-6, None)).mean(axis=1)
           if big.any() else np.full(len(mu), np.nan))
    log_esp = np.asarray(g.esp, float)
    return {"cid": path.stem, "ideal": ideal, "att": att,
            "e": -log_esp, "esp_missing": int(np.sum(log_esp == 0.0)),
            "x": np.array([[float(r[f]) for f in FEATS] for r in d["records"]]),
            "g2": np.array([gamma2(r) for r in d["records"]]),
            "bv": bias_variance(d)}


def pooled(C, idx):
    """Stack per-circuit demeaned attenuation, ESP error mass and features."""
    A, E, X = [], [], []
    for i in idx:
        c = C[i]
        A.append(c["att"] - c["att"].mean())
        E.append(c["e"] - c["e"].mean())
        X.append(c["x"] - c["x"].mean(axis=0))
    return np.concatenate(A), np.concatenate(E)[:, None], np.concatenate(X)


def r2_out_of_fold(C, folds, use_feats):
    """Fit on one half of the circuits, score on the other half, and swap."""
    ss_res = ss_tot = 0.0
    for k in (0, 1):
        A_tr, E_tr, X_tr = pooled(C, folds[1 - k])
        A_te, E_te, X_te = pooled(C, folds[k])
        Z_tr = np.hstack([E_tr, X_tr]) if use_feats else E_tr
        Z_te = np.hstack([E_te, X_te]) if use_feats else E_te
        coef = np.linalg.lstsq(Z_tr, A_tr, rcond=None)[0]
        ss_res += np.sum((A_te - Z_te @ coef) ** 2)
        ss_tot += np.sum(A_te ** 2)
    return 1.0 - ss_res / ss_tot


def main():
    assert os.environ.get("QUEST_UNBLIND") == "holdout3", "set QUEST_UNBLIND=holdout3"
    C = []
    for p in sorted((DATA_V2 / "holdout3").glob("*.json")):
        if "g1_empty" in p.name:
            continue
        c = load(p)
        if c is not None:
            C.append(c)
    missing = sum(c["esp_missing"] for c in C)
    print(f"circuits: {len(C)}  candidates: {sum(len(c['e']) for c in C)}  "
          f"candidates without layout-ESP: {missing}")
    assert missing == 0, "some candidates have no layout-ESP; stop and inspect"

    use = [i for i, c in enumerate(C) if len(c["e"]) >= 3 and np.all(np.isfinite(c["att"]))]
    perm = np.random.default_rng(SPLIT_SEED).permutation(use)
    folds = (perm[0::2], perm[1::2])

    print("\n=== Q1. how much of the attenuation does layout-ESP explain (within circuit)? ===")
    rho = np.array([spearmanr(C[i]["e"], C[i]["att"])[0] for i in use])
    rho = rho[np.isfinite(rho)]
    A, E, X = pooled(C, use)
    print(f"circuits used: {len(use)}")
    print(f"per-circuit Spearman(-log ESP, attenuation): median {np.median(rho):+.3f}, "
          f"share > 0: {np.mean(rho > 0):.3f}")
    print(f"pooled within-circuit R^2 (in-sample): {np.corrcoef(A, E[:, 0])[0, 1] ** 2:.3f}")

    print("\n=== Q2. is the remainder learnable? (2-fold by circuit, out-of-fold) ===")
    r2_esp = r2_out_of_fold(C, folds, False)
    r2_all = r2_out_of_fold(C, folds, True)
    print(f"R^2 ESP only: {r2_esp:.3f}   ESP + cut features: {r2_all:.3f}   "
          f"gain: {r2_all - r2_esp:+.3f}")
    Z = np.hstack([E, X])
    s = Z.std(axis=0)
    s[s == 0] = 1.0
    coef = np.linalg.lstsq((Z - Z.mean(axis=0)) / s, A / A.std(), rcond=None)[0]
    print("standardised coefficients (in-sample, descriptive only):")
    for name, b in sorted(zip(("-log ESP",) + FEATS, coef), key=lambda t: -abs(t[1])):
        print(f"  {name:>24}: {b:+.3f}")

    print("\n=== Q3. budget-aware selection: geometric mean of MSE / oracle MSE ===")
    rules = ("min_overhead", "max_esp", "esp_budget")
    ratios = {r: {N: [] for N in BUDGETS} for r in rules}
    for k in (0, 1):
        tr = folds[1 - k]
        slope, icpt = np.polyfit(np.concatenate([C[i]["e"] for i in tr]),
                                 np.concatenate([C[i]["att"] for i in tr]), 1)
        print(f"fold {k}: attenuation ~ {icpt:+.4f} + {slope:+.4f} * (-log ESP)")
        for i in folds[k]:
            c = C[i]
            b2, v = c["bv"][:, 0], c["bv"][:, 1]
            f_hat = np.exp(-(icpt + slope * c["e"]))
            pred = f_hat[:, None] * c["ideal"][None, :]
            b2_hat = np.mean((pred - c["ideal"][None, :]) ** 2, axis=1)
            v_hat = np.mean(np.clip(c["g2"][:, None] - pred ** 2, 0.0, None), axis=1)
            for N in BUDGETS:
                true = b2 + v / N
                best = max(true.min(), 1e-300)
                picks = {"min_overhead": int(np.lexsort((c["e"], c["g2"]))[0]),
                         "max_esp": int(np.argmin(c["e"])),
                         "esp_budget": int(np.argmin(b2_hat + v_hat / N))}
                for r in rules:
                    ratios[r][N].append(true[picks[r]] / best)
    print(f"{'budget':>8} " + " ".join(f"{r:>13}" for r in rules) + "   esp_budget within 10%")
    for N in BUDGETS:
        row = [np.exp(np.mean(np.log(ratios[r][N]))) for r in rules]
        w10 = np.mean(np.array(ratios["esp_budget"][N]) <= 1.10)
        print(f"{N:>8.0e} " + " ".join(f"{x:>13.3f}" for x in row) + f"   {w10:.3f}")


if __name__ == "__main__":
    main()
