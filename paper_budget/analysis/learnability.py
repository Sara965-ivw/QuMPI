"""Nonlinear learnability of the cut-induced attenuation (exploratory; holdout3 v3.1).

Does a gradient-boosting model on ESP and simple cut features predict the attenuation,
out of fold by circuit, well enough to improve budget-aware selection?
Also: the ceiling of any predictor that outputs ONE attenuation per candidate.
Hyperparameters are fixed in advance (no tuning).
Predictions were committed beforehand in notes/PREDICTIONS_learnability.md.
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from paper_budget.config import DATA_V2, BUDGETS
from paper_budget.analysis.headroom import load

FAMS = ("clustered_random", "qaoa_sbm", "twolocal")
N_FOLDS, GBM_SEED = 5, 2026
GBM = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
           l2_regularization=1.0, random_state=GBM_SEED)


def design(c, meta, with_cut):
    """Per-candidate inputs: ESP (raw and within-circuit), circuit context, cut features."""
    K = len(c["e"])
    ctx = np.column_stack([np.full(K, float(meta["n_qubits"]))]
                          + [np.full(K, float(meta["family"] == f)) for f in FAMS])
    cols = [c["e"][:, None], (c["e"] - c["e"].mean())[:, None], ctx]
    if with_cut:
        cols += [c["x"], c["x"] - c["x"].mean(axis=0)]
    return np.hstack(cols)


def main():
    assert os.environ.get("QUEST_UNBLIND") == "holdout3", "set QUEST_UNBLIND=holdout3"
    C, M = [], []
    for p in sorted((DATA_V2 / "holdout3").glob("*.json")):
        if "g1_empty" in p.name:
            continue
        c = load(p)
        if c is not None and len(c["e"]) >= 3 and np.all(np.isfinite(c["att"])):
            C.append(c)
            M.append(json.loads(p.read_text())["circuit_meta"])
    rows = np.concatenate([np.full(len(c["e"]), i) for i, c in enumerate(C)])
    y = np.concatenate([c["att"] for c in C])
    X = {w: np.vstack([design(c, m, w) for c, m in zip(C, M)]) for w in (False, True)}
    e_all = np.concatenate([c["e"] for c in C])
    print(f"circuits: {len(C)}  candidates: {len(y)}  folds: {N_FOLDS} (grouped by circuit)")

    pred = {"linear_esp": np.zeros_like(y), "gbm_esp": np.zeros_like(y),
            "gbm_esp_cut": np.zeros_like(y)}
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(y, y, rows):
        slope, icpt = np.polyfit(e_all[tr], y[tr], 1)
        pred["linear_esp"][te] = icpt + slope * e_all[te]
        for name, w in (("gbm_esp", False), ("gbm_esp_cut", True)):
            model = HistGradientBoostingRegressor(**GBM).fit(X[w][tr], y[tr])
            pred[name][te] = model.predict(X[w][te])

    print("\n=== A1. out-of-fold prediction of the attenuation (within circuit) ===")
    for name, p in pred.items():
        t_dm = np.concatenate([y[rows == i] - y[rows == i].mean() for i in range(len(C))])
        p_dm = np.concatenate([p[rows == i] - p[rows == i].mean() for i in range(len(C))])
        r2 = 1.0 - np.sum((t_dm - p_dm) ** 2) / np.sum(t_dm ** 2)
        rho = np.array([spearmanr(p[rows == i], y[rows == i])[0] for i in range(len(C))])
        rho = rho[np.isfinite(rho)]
        print(f"{name:>12}: within-circuit R^2 {r2:+.3f}   median Spearman {np.median(rho):+.3f}")

    print("\n=== A2. budget-aware selection: geometric mean of MSE / oracle MSE ===")
    rules = ["min_overhead"] + [f"{n}_budget" for n in pred] + ["true_att_budget"]
    ratios = {r: {N: [] for N in BUDGETS} for r in rules}
    for i, c in enumerate(C):
        b2, v = c["bv"][:, 0], c["bv"][:, 1]
        att_by_rule = {f"{n}_budget": p[rows == i] for n, p in pred.items()}
        att_by_rule["true_att_budget"] = c["att"]
        scores = {}
        for r, att in att_by_rule.items():
            mu_hat = np.exp(-att)[:, None] * c["ideal"][None, :]
            scores[r] = (np.mean((mu_hat - c["ideal"][None, :]) ** 2, axis=1),
                         np.mean(np.clip(c["g2"][:, None] - mu_hat ** 2, 0.0, None), axis=1))
        for N in BUDGETS:
            true = b2 + v / N
            best = max(true.min(), 1e-300)
            ratios["min_overhead"][N].append(true[np.lexsort((c["e"], c["g2"]))[0]] / best)
            for r, (b2_hat, v_hat) in scores.items():
                ratios[r][N].append(true[int(np.argmin(b2_hat + v_hat / N))] / best)
    print(f"{'budget':>8} " + " ".join(f"{r:>20}" for r in rules))
    for N in BUDGETS:
        row = [np.exp(np.mean(np.log(ratios[r][N]))) for r in rules]
        print(f"{N:>8.0e} " + " ".join(f"{x:>20.3f}" for x in row))


if __name__ == "__main__":
    main()
