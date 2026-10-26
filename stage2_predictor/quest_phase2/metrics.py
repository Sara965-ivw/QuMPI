"""Per-circuit ranking metrics, bootstrap confidence intervals and paired tests.

A metric is computed within each circuit (the decision RQ1 cares about is
which cut to choose for THIS circuit), then averaged over circuits.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def circuit_metrics(y: np.ndarray, s: np.ndarray, k: int = 3) -> dict:
    """y: true labels (higher is better), s: model scores (higher is better)."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    out = {"n_cand": len(y)}
    if len(y) < 2:
        return out
    best = int(np.argmax(s))
    out["top1_regret"] = float(y.max() - y[best])                 # fidelity lost by the pick
    out["top1_hit"] = float(y[best] >= y.max() - 1e-12)
    if np.ptp(y) > 0:
        if np.ptp(s) > 0:
            out["kendall"] = float(stats.kendalltau(y, s).statistic)
            out["spearman"] = float(stats.spearmanr(y, s).statistic)
        else:                      # constant scores carry no ranking: count as 0, never drop
            out["kendall"] = out["spearman"] = 0.0
        out["kendall_w"] = kendall_gap_weighted(y, s)
    # NDCG@k with gains = labels rescaled to [0, 1] within the circuit
    g = (y - y.min()) / (np.ptp(y) or 1.0)
    order = np.argsort(-s)[:k]
    ideal = np.sort(g)[::-1][:k]
    disc = 1.0 / np.log2(np.arange(2, len(order) + 2))
    idcg = float((ideal * disc[:len(ideal)]).sum())
    out["ndcg3"] = float((g[order] * disc).sum() / idcg) if idcg > 0 else 1.0
    return out


METRICS = ["kendall", "kendall_w", "spearman", "top1_regret", "top1_hit", "ndcg3"]


def kendall_gap_weighted(y, s) -> float:
    """Pre-registered primary metric for holdout3 (PREREG_holdout3.md, Section 5):
    each candidate pair is weighted by its label gap |y_i - y_j|; +1 for a pair
    ordered correctly, -1 for a reversed pair, 0 for a tie in the scores.
    Pairs whose labels differ by almost nothing barely count."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    i, j = np.triu_indices(len(y), 1)
    w = np.abs(y[i] - y[j])
    if w.sum() <= 0:
        return 0.0
    agree = np.sign(y[i] - y[j]) * np.sign(s[i] - s[j])
    return float((w * agree).sum() / w.sum())


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple:
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], float)
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarise(rows: list, group_keys=("model", "split")) -> list:
    """rows: per-circuit dicts with model, split, family, n_qubits, metrics."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)
    out = []
    for key, rs in sorted(groups.items()):
        rec = dict(zip(group_keys, key))
        rec["n_circuits"] = len({r["cid"] for r in rs})
        for m in METRICS:
            mean, lo, hi = bootstrap_ci([r.get(m) for r in rs])
            rec[m], rec[m + "_lo"], rec[m + "_hi"] = mean, lo, hi
        out.append(rec)
    return out


def paired_test(rows: list, model_a: str, model_b: str, split: str, metric: str = "kendall") -> dict:
    """Wilcoxon signed-rank test on per-circuit metric values (circuits paired),
    with the rank-biserial correlation as effect size. Per-circuit values are
    first averaged over seeds so each circuit contributes one pair."""
    def per_circuit(model):
        acc = {}
        for r in rows:
            if r["model"] == model and r["split"] == split and r.get(metric) is not None:
                acc.setdefault(r["cid"], []).append(r[metric])
        return {c: float(np.mean(v)) for c, v in acc.items()}
    a, b = per_circuit(model_a), per_circuit(model_b)
    common = sorted(set(a) & set(b))
    d = np.array([a[c] - b[c] for c in common])
    if len(common) < 5 or np.allclose(d, 0):
        return {"n": len(common), "p": float("nan"), "effect_rb": float("nan"),
                "mean_diff": float(d.mean()) if len(d) else float("nan")}
    w = stats.wilcoxon(d, zero_method="wilcox")
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    return {"n": len(common), "p": float(w.pvalue), "effect_rb": rb, "mean_diff": float(d.mean())}
