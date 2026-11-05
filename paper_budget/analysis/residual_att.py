"""Residual attenuation model: ESP prior plus a learned within-circuit correction (exploratory; holdout3).

learnability.py trained GBM on the raw attenuation and got negative within-circuit R^2 (design flaw).
Here the GBM learns only what the linear ESP fit misses, within each circuit:
    att_hat = lin(e)  +  gbm(x) - mean_circuit(gbm(x))
so with no signal the correction is zero and the model falls back to ESP (the residual / zero-init idea).
Selection uses the true m = mean mu*^2 for every model, to isolate the attenuation estimate
(deployable_rule.py showed m costs at most a few percent). Folds: GroupKFold(5) by circuit, as in
learnability.py; the linear fit is refitted inside each fold. GBM settings are learnability.GBM (no tuning).
Predictions were committed beforehand in notes/PREDICTIONS_residual_att.md.
"""
import json
import os

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from paper_budget.config import DATA_V2, BUDGETS
from paper_budget.analysis.headroom import load
from paper_budget.analysis.learnability import design, GBM, N_FOLDS
from paper_budget.analysis.deployable_rule import choose, holm

N_ARR = np.asarray(BUDGETS, float)


def demean(v, rows):
    out = np.empty_like(v)
    for i in np.unique(rows):
        out[rows == i] = v[rows == i] - v[rows == i].mean()
    return out


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
    e = np.concatenate([c["e"] for c in C])
    X = np.vstack([design(c, m, True) for c, m in zip(C, M)])
    Xc = np.vstack([np.hstack([c["e"][:, None] - c["e"].mean(), c["x"] - c["x"].mean(axis=0)]) for c in C])
    print(f"circuits: {len(C)}  candidates: {len(y)}  folds: {N_FOLDS} (grouped by circuit)")

    pred = {k: np.zeros_like(y) for k in ("lin_esp", "lin_esp_feats", "gbm_resid")}
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(y, y, rows):
        b, a = np.polyfit(e[tr], y[tr], 1)
        lin_tr, lin_te = a + b * e[tr], a + b * e[te]
        res_tr = demean(y[tr] - lin_tr, rows[tr])                 # what ESP misses, within circuit
        pred["lin_esp"][te] = lin_te
        coef = np.linalg.lstsq(Xc[tr], res_tr, rcond=None)[0]     # linear correction (no intercept)
        pred["lin_esp_feats"][te] = lin_te + demean(Xc[te] @ coef, rows[te])
        gbm = HistGradientBoostingRegressor(**GBM).fit(X[tr], res_tr)
        pred["gbm_resid"][te] = lin_te + demean(gbm.predict(X[te]), rows[te])

    print("\n=== R1. out-of-fold within-circuit R^2 of the attenuation ===")
    t_dm = demean(y, rows)
    for k, p in pred.items():
        p_dm = demean(p, rows)
        rho = [spear for spear in (stats.spearmanr(p[rows == i], y[rows == i])[0] for i in range(len(C)))
               if np.isfinite(spear)]
        print(f"{k:>14}: R^2 {1 - np.sum((t_dm - p_dm) ** 2) / np.sum(t_dm ** 2):+.3f}   "
              f"median Spearman {np.median(rho):+.3f}")

    rules = ["min_overhead"] + list(pred) + ["ceiling_true_att"]
    R = {r: np.zeros((len(C), len(BUDGETS))) for r in rules}
    for i, c in enumerate(C):
        b2, v, g2 = c["bv"][:, 0], c["bv"][:, 1], c["g2"]
        m = float(np.mean(c["ideal"] ** 2))
        att = {k: p[rows == i] for k, p in pred.items()}
        att["ceiling_true_att"] = c["att"]
        mo = int(np.lexsort((c["e"], g2))[0])
        for j, N in enumerate(N_ARR):
            true = b2 + v / N
            best = max(true.min(), 1e-300)
            R["min_overhead"][i, j] = np.log(true[mo] / best)
            for k, a_k in att.items():
                R[k][i, j] = np.log(true[choose(np.exp(-a_k), g2, m, N)] / best)

    print("\n=== R2. geometric mean of MSE / oracle MSE (selection with the true m) ===")
    print(f"{'budget':>8} " + " ".join(f"{r:>17}" for r in rules) + "   gap closed")
    for j, N in enumerate(N_ARR):
        g = {r: np.exp(R[r][:, j].mean()) for r in rules}
        gap = g["lin_esp"] - g["ceiling_true_att"]
        closed = (g["lin_esp"] - g["gbm_resid"]) / gap if gap > 1e-3 else float("nan")
        print(f"{N:>8.0e} " + " ".join(f"{g[r]:>17.3f}" for r in rules) + f"   {closed:>9.2f}")

    print("\n=== R3. paired: geometric-mean MSE ratio reference / model (> 1 = model better), Holm ===")
    for ref, mod in (("lin_esp", "gbm_resid"), ("lin_esp", "lin_esp_feats"), ("min_overhead", "gbm_resid")):
        L = R[ref] - R[mod]
        ps = [float(stats.wilcoxon(L[:, j]).pvalue) if np.any(np.abs(L[:, j]) > 1e-12) else 1.0
              for j in range(len(BUDGETS))]
        print(f"{ref + ' / ' + mod:<28}" + "".join(f"{np.exp(L[:, j].mean()):>7.3f}({q:.0e})"
                                                  for j, q in enumerate(holm(ps))))


if __name__ == "__main__":
    main()
