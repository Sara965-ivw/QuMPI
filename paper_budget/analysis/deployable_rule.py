"""Budget-aware selection without the ideal values (exploratory; holdout3, scheme v3.1).

headroom.py Q3 and learnability.py A2 chose cuts with
    b2_hat_k + v_hat_k / N,   mu_hat_kS = f_hat_k * mu*_S,
i.e. they were given the ideal expectation values mu*_S, which a user does not have.
With one attenuation f_k per cut the rule depends on mu* only through ONE number,
m = mean_S mu*_S^2:
    b2_hat_k = (1 - f_k)^2 m,     v_hat_k = gamma2_k - f_k^2 m      (when no clipping)
This script replaces m by quantities a user can obtain and measures the loss:
    m_true    m itself (must equal the per-observable rule up to clipping: check C2)
    m_one     worst case m = 1 (all |mu*| = 1)
    m_x0.25, m_x4   m off by a factor of 4 (sensitivity)
    m_pilot   10% of the shots run on the minimum-overhead cut, m estimated from them,
              the remaining 90% on the chosen cut (pilot shots are not reused: conservative)
Attenuation f_hat: the out-of-fold linear ESP fit of headroom.py Q3 (same folds), so that
obs_true reproduces the Q3 'esp_budget' column (check C1).
Predictions were committed beforehand in notes/PREDICTIONS_deployable_rule.md.
"""
import json
import os

import numpy as np
from scipy import stats

from paper_budget.config import DATA_V2, BUDGETS, SPLIT_SEED
from paper_budget.analysis.headroom import load

PILOT = 0.10        # share of the budget spent on the pilot
REPEATS = 20        # simulated pilots per (circuit, budget)
N_ARR = np.asarray(BUDGETS, float)


def choose(f, g2, m, N):
    """Index of the cut with the lowest predicted MSE when mean mu*^2 = m."""
    b2 = (1.0 - f) ** 2 * m
    v = np.clip(g2 - f ** 2 * m, 0.0, None)
    return int(np.argmin(b2 + v / N))


def choose_obs(f, g2, ideal, N):
    """Per-observable form used in headroom.py Q3 (needs every mu*_S)."""
    pred = f[:, None] * ideal[None, :]
    b2 = np.mean((pred - ideal[None, :]) ** 2, axis=1)
    v = np.mean(np.clip(g2[:, None] - pred ** 2, 0.0, None), axis=1)
    return int(np.argmin(b2 + v / N))


def pilot_m(mu_mo, g2_mo, f_mo, n_p, rng, repeats):
    """Estimate m from n_p shots on the minimum-overhead cut (Gaussian approximation).

    Each observable's estimate has mean mu_mo,S and variance (gamma2 - mu^2) / n_p.
    E[mu_hat^2] = mu^2 (1 - 1/n_p) + gamma2 / n_p, so (mu_hat^2 - gamma2/n_p) / (1 - 1/n_p)
    is unbiased for mu^2; dividing by f_mo^2 undoes the attenuation of that cut.
    """
    sd = np.sqrt(np.clip(g2_mo - mu_mo ** 2, 0.0, None) / n_p)
    mu_hat = mu_mo[None, :] + sd[None, :] * rng.standard_normal((repeats, len(mu_mo)))
    mu2 = (mu_hat ** 2 - g2_mo / n_p) / (1.0 - 1.0 / n_p)
    return np.clip(mu2.mean(axis=1) / f_mo ** 2, 0.0, 1.0)


def gm(x):
    return float(np.exp(np.mean(np.log(x))))


def holm(ps):
    ps = np.asarray(ps, float)
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def main():
    assert os.environ.get("QUEST_UNBLIND") == "holdout3", "set QUEST_UNBLIND=holdout3"
    C, MU = [], []
    for p in sorted((DATA_V2 / "holdout3").glob("*.json")):
        if "g1_empty" in p.name:
            continue
        c = load(p)
        if c is not None:
            C.append(c)
            MU.append(np.array([r["obs_mu"] for r in json.loads(p.read_text())["records"]], float))
    use = [i for i, c in enumerate(C) if len(c["e"]) >= 3 and np.all(np.isfinite(c["att"]))]
    perm = np.random.default_rng(SPLIT_SEED).permutation(use)
    folds = (perm[0::2], perm[1::2])                      # identical to headroom.py

    rules = ["min_overhead", "max_esp", "obs_true", "m_true", "m_one", "m_x0.25", "m_x4",
             "m_pilot", "ceiling_true_att"]
    R = {r: np.zeros((len(use), len(BUDGETS))) for r in rules}   # log(MSE / oracle)
    m_info, n_diff, n_row = [], 0, 0
    row = 0
    for k in (0, 1):
        tr = folds[1 - k]
        slope, icpt = np.polyfit(np.concatenate([C[i]["e"] for i in tr]),
                                 np.concatenate([C[i]["att"] for i in tr]), 1)
        print(f"fold {k}: attenuation ~ {icpt:+.4f} + {slope:+.4f} * (-log ESP)")
        for i in folds[k]:
            c, mu = C[i], MU[i]
            b2, v, g2, ideal = c["bv"][:, 0], c["bv"][:, 1], c["g2"], c["ideal"]
            f = np.exp(-(icpt + slope * c["e"]))
            f_true = np.exp(-c["att"])
            m = float(np.mean(ideal ** 2))
            mo = int(np.lexsort((c["e"], g2))[0])
            m_info.append(m)
            for j, N in enumerate(N_ARR):
                true = b2 + v / N
                best = max(true.min(), 1e-300)
                picks = {"min_overhead": mo, "max_esp": int(np.argmin(c["e"])),
                         "obs_true": choose_obs(f, g2, ideal, N),
                         "m_true": choose(f, g2, m, N),
                         "m_one": choose(f, g2, 1.0, N),
                         "m_x0.25": choose(f, g2, min(m * 0.25, 1.0), N),
                         "m_x4": choose(f, g2, min(m * 4.0, 1.0), N),
                         "ceiling_true_att": choose_obs(f_true, g2, ideal, N)}
                n_diff += picks["obs_true"] != picks["m_true"]
                n_row += 1
                for r, kk in picks.items():
                    R[r][row, j] = np.log(true[kk] / best)
                # pilot: n_p shots on the minimum-overhead cut, the rest on the chosen cut
                n_p = max(int(round(PILOT * N)), 2)
                rng = np.random.default_rng([SPLIT_SEED, int(i), j])
                m_hat = pilot_m(mu[mo], g2[mo], f[mo], n_p, rng, REPEATS)
                rest = N - n_p
                logs = [np.log((b2[kk] + v[kk] / rest) / best)
                        for kk in (choose(f, g2, mh, rest) for mh in m_hat)]
                R["m_pilot"][row, j] = np.mean(logs)
            row += 1

    m_info = np.array(m_info)
    print(f"\ncircuits: {len(use)}   mean mu*^2 per circuit: median {np.median(m_info):.3f}, "
          f"IQR [{np.quantile(m_info, .25):.3f}, {np.quantile(m_info, .75):.3f}]")
    print(f"C2 obs_true and m_true choose different cuts in {n_diff} of {n_row} (circuit, budget) cases "
          f"(should be 0 unless clipping)")
    print("\nGeometric mean of MSE / oracle MSE (oracle = best candidate at the full budget N)")
    print(f"{'budget':>8} " + " ".join(f"{r:>16}" for r in rules))
    for j, N in enumerate(N_ARR):
        print(f"{N:>8.0e} " + " ".join(f"{np.exp(R[r][:, j].mean()):>16.3f}" for r in rules))

    print("\nPaired, per circuit: geometric-mean MSE ratio reference / m_pilot (> 1 = m_pilot better), "
          "Wilcoxon, Holm over budgets")
    for ref in ("min_overhead", "max_esp", "m_true"):
        L = R[ref] - R["m_pilot"]
        ps = [float(stats.wilcoxon(L[:, j]).pvalue) if np.any(np.abs(L[:, j]) > 1e-12) else 1.0
              for j in range(len(BUDGETS))]
        print(f"{ref + ' / m_pilot':<26}" + "".join(f"{np.exp(L[:, j].mean()):>7.3f}({q:.0e})"
                                                  for j, q in enumerate(holm(ps))))
    print(f"\nm_pilot / m_true per budget (prediction P4): "
          + " ".join(f"{np.exp((R['m_pilot'][:, j] - R['m_true'][:, j]).mean()):.3f}"
                     for j in range(len(BUDGETS))))


if __name__ == "__main__":
    main()
