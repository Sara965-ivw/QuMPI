"""Per-observable headroom (exploratory; holdout3, scheme v3.1).

Is the cut that is best for the whole observable basket also the best cut for each single
observable? If not, a model that is told which observable matters (observable-conditioned
prediction) could gain more than the basket-level ceiling measured in learnability.py.
MSE of candidate k for observable S at N shots: (mu_kS - mu*_S)^2 + max(gamma2_k - mu_kS^2, 0) / N,
the per-observable form of paper_budget.targets (whose basket MSE is its mean over S).
Descriptive only: pairs share circuits, so no p-values are given.
Predictions were committed beforehand in notes/PREDICTIONS_per_observable.md.
"""
import json
import os

import numpy as np

from paper_budget.config import DATA_V2, BUDGETS
from paper_budget.analysis.headroom import load

BIG = 0.2          # informative observables: |mu*| > 0.2 (they carry most of the error)


def gmean(x):
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def main():
    assert os.environ.get("QUEST_UNBLIND") == "holdout3", "set QUEST_UNBLIND=holdout3"
    res = {N: {"basket": [], "minover": [], "circ_basket": []} for N in BUDGETS}
    n_circ = n_pairs = 0
    for p in sorted((DATA_V2 / "holdout3").glob("*.json")):
        if "g1_empty" in p.name:
            continue
        c = load(p)
        if c is None or len(c["e"]) < 3:
            continue
        ideal = c["ideal"]
        inf = np.abs(ideal) > BIG
        if not inf.any():
            continue
        mu = np.array([r["obs_mu"] for r in json.loads(p.read_text())["records"]], float)  # (K, S)
        b2 = (mu - ideal[None, :]) ** 2
        v = np.clip(c["g2"][:, None] - mu ** 2, 0.0, None)
        mo = int(np.lexsort((c["e"], c["g2"]))[0])          # minimum overhead, ties by ESP
        n_circ += 1
        n_pairs += int(inf.sum())
        for N in BUDGETS:
            m = b2 + v / N                                   # (K, S)
            k_basket = int(np.argmin(m.mean(axis=1)))        # best cut for the basket
            best = np.maximum(m.min(axis=0), 1e-300)         # best cut for each observable
            r_basket = (m[k_basket] / best)[inf]
            r_minover = (m[mo] / best)[inf]
            res[N]["basket"].extend(r_basket.tolist())
            res[N]["minover"].extend(r_minover.tolist())
            res[N]["circ_basket"].append(gmean(r_basket))

    print(f"circuits: {n_circ}   informative (circuit, observable) pairs: {n_pairs}")
    print("\nRatio = MSE of a rule's cut for that observable / MSE of the best cut for that observable")
    print(f"{'budget':>8} {'basket-best cut':>16} {'share > 1.10':>13} {'circuits > 1.10':>16} "
          f"{'min-overhead cut':>17} {'share > 1.10':>13}")
    for N in BUDGETS:
        rb, rm, cb = map(np.array, (res[N]["basket"], res[N]["minover"], res[N]["circ_basket"]))
        print(f"{N:>8.0e} {gmean(rb):>16.3f} {np.mean(rb > 1.10):>13.3f} {np.mean(cb > 1.10):>16.3f} "
              f"{gmean(rm):>17.3f} {np.mean(rm > 1.10):>13.3f}")


if __name__ == "__main__":
    main()
