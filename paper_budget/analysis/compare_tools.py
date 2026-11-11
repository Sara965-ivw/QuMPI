"""Labelled comparison: our selection rules vs IBM find_cuts vs Qdislib (exploratory, holdout3).

Circuits: the find_cuts comparison set (runs/findcuts_compare/per_circuit.csv) minus circuits
excluded there; method picks (model, GBM, layout-ESP, minimum overhead, METIS) are read from
that file, so they are the frozen October picks. Every cut is scored with the same exact
finite-shot MSE as paper_budget.targets, at 1e3-1e9 shots:
    MSE_k(N) = mean_S (mu_kS - mu*_S)^2 + mean_S max(gamma2_k - mu_kS^2, 0) / N
Qdislib (default find_cut, 5 seeds averaged): a seed whose cut cannot be labelled
(unrepresentable or too many gate cuts) gets the optimistic bound (gamma2 - 1) / N (zero bias),
decision Q3; a circuit with a seed that found no feasible cut is listed and left out of the
Qdislib tests. Oracle = best cut among the candidates, the find_cuts cut and the labelled
Qdislib cuts. Paired tests per circuit: log MSE ratio, Wilcoxon, Holm over the 7 budgets.
Predictions L1-L3: notes/PREDICTIONS_qdislib.md (committed before the Qdislib run).
"""
import csv
import json
import math
import os

import numpy as np
from scipy import stats

from paper_budget.config import DATA_V2, RUNS, ROOT, BUDGETS
from paper_budget.targets import bias_variance, gamma2

FC_CSV = ROOT / "runs" / "findcuts_compare" / "per_circuit.csv"
FC_LAB = ROOT / "runs" / "findcuts_label_v2"
QD_SEL = RUNS / "qdislib" / "select"
QD_LAB = RUNS / "qdislib" / "labels"
OUT = RUNS / "compare"
SEEDS = ("1234", "2027", "3141", "4242", "5555")
PICKED = ("model", "gbm-layout", "esp_layout", "min_overhead", "metis")
N = np.asarray(BUDGETS, float)
TIE = 1e-12


def mse_of(bv):
    """(K, 2) [b2, v] -> (K, len(BUDGETS)) MSE."""
    return bv[:, :1] + bv[:, 1:2] / N[None, :]


def label_mse(lab, ideal):
    rec = {"obs_mu": lab["obs_mu"], "gamma2": lab["gamma2"]}
    return mse_of(bias_variance({"obs_ideal": ideal, "records": [rec]}))[0]


def holm(ps):
    ps = np.asarray(ps, float)
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def wilcoxon_p(d):
    d = np.asarray(d, float)
    return float(stats.wilcoxon(d).pvalue) if len(d) >= 5 and np.any(np.abs(d) > TIE) else float("nan")


def qdislib_cut(cid, ideal, sel_dir=None, lab_dir=None):
    """Seed-averaged (MSE vector, fidelity or nan, bound used?) or (None, reason).
    sel_dir / lab_dir default to the October holdout3 folders (QD_SEL, QD_LAB)."""
    sel = json.loads(((sel_dir or QD_SEL) / f"{cid}.json").read_text())
    labs = json.loads(((lab_dir or QD_LAB) / f"{cid}.json").read_text())
    if labs["status"] != "done":
        return None, f"labelling {labs['status']}"
    rows, fids, bound = [], [], False
    for s in SEEDS:
        v = sel["default"][s]
        if v["status"] == "ok":
            e = next(c for c in labs["cuts"] if f"default_{s}" in c["used_by"])
            rows.append(label_mse(e["label"], ideal))
            fids.append(e["label"]["pst_hellinger"])
        elif v["status"] in ("unrepresentable", "too_many_cuts"):
            rows.append(np.maximum(2.0 ** v["log2_gamma2"] - 1.0, 0.0) / N)
            bound = True
        else:
            return None, f"seed {s}: {v['status']}"
    return (np.mean(rows, axis=0), float(np.mean(fids)) if len(fids) == len(SEEDS) else float("nan"), bound), None


def main():
    assert os.environ.get("QUEST_UNBLIND") == "holdout3", "set QUEST_UNBLIND=holdout3"
    OUT.mkdir(parents=True, exist_ok=True)
    fc_rows = {r["cid"]: r for r in csv.DictReader(open(FC_CSV))}
    rows, skipped = [], []
    for cid, r in sorted(fc_rows.items()):
        if r.get("excluded"):
            skipped.append((cid, r["excluded"]))
            continue
        d = json.loads((DATA_V2 / "holdout3" / f"{cid}.json").read_text())
        ideal = d["obs_ideal"]
        assert int(r["n_cand"]) == len(d["records"]), f"{cid}: candidate count differs from October"
        M = mse_of(bias_variance(d))                                   # (K, 7)
        F = np.array([x["pst_hellinger"] for x in d["records"]], float)
        row = {"cid": cid, "family": r["family"], "n_qubits": int(r["n_qubits"]),
               "separable": r["separable"] == "True"}
        meth = {}
        for name in PICKED:
            picks = json.loads(r[f"{name}_picks"])
            meth[name] = (M[picks].mean(0), float(F[picks].mean()))
        meth["random"] = (M.mean(0), float(F.mean()))
        fl = json.loads((FC_LAB / f"{cid}.json").read_text())
        assert fl["status"] == "ok" and fl["label"] is not None, f"{cid}: find_cuts label missing"
        meth["find_cuts"] = (label_mse(fl["label"], ideal), float(fl["label"]["pst_hellinger"]))
        qd, why = qdislib_cut(cid, ideal)
        if qd is None:
            row["qdislib_missing"] = why
        else:
            meth["qdislib"] = (qd[0], qd[1])
            row["qdislib_bound"] = qd[2]
        qd_cuts = [label_mse(c["label"], ideal) for c in
                   json.loads((QD_LAB / f"{cid}.json").read_text())["cuts"] if c["label"]]
        oracle_cand = M.min(0)
        oracle_fc = np.minimum(oracle_cand, meth["find_cuts"][0])
        oracle_qd = np.minimum(oracle_cand, np.min(qd_cuts, axis=0)) if qd_cuts else oracle_cand
        oracle = np.minimum(oracle_fc, oracle_qd)
        row.update(oracle_cand=oracle_cand, oracle_fc=oracle_fc, oracle_qd=oracle_qd, oracle=oracle)
        for name, (m, f) in meth.items():
            row[f"{name}|mse"], row[f"{name}|F"] = m, f
        rows.append(row)

    methods = ["model", "gbm-layout", "esp_layout", "min_overhead", "metis", "random", "find_cuts", "qdislib"]
    subsets = {"all": rows, "connected": [r for r in rows if not r["separable"]],
               "separable": [r for r in rows if r["separable"]]}
    print(f"circuits: {len(rows)} (skipped as in October: {skipped})")
    print(f"Qdislib missing: {[(r['cid'], r['qdislib_missing']) for r in rows if 'qdislib_missing' in r]}")
    print(f"Qdislib with the optimistic bound on some seed: {sum(bool(r.get('qdislib_bound')) for r in rows)}")
    report = {}
    for sname, rs in subsets.items():
        print(f"\n== {sname} (n = {len(rs)}): geometric mean of MSE / oracle (oracle includes find_cuts and Qdislib cuts)")
        print(f"{'method':<13}{'F':>8}" + "".join(f"{f'1e{int(math.log10(b))}':>8}" for b in BUDGETS))
        for mname in methods:
            have = [r for r in rs if f"{mname}|mse" in r]
            ratios = np.array([r[f"{mname}|mse"] / r["oracle"] for r in have])
            fid = [r[f"{mname}|F"] for r in have if np.isfinite(r[f"{mname}|F"])]
            g = np.exp(np.log(ratios).mean(0)) if len(have) else [float("nan")] * len(BUDGETS)
            print(f"{mname:<13}{np.mean(fid) if fid else float('nan'):>8.4f}" + "".join(f"{x:>8.2f}" for x in g)
                  + ("" if len(have) == len(rs) else f"   (n = {len(have)})"))

        print(f"-- paired: geometric-mean MSE ratio reference / method (> 1 = method better), Holm over budgets")
        report[sname] = {}
        for ref in ("find_cuts", "qdislib"):
            for mname in ["model", "gbm-layout", "esp_layout", "min_overhead"] + (["find_cuts"] if ref == "qdislib" else []):
                both = [r for r in rs if f"{ref}|mse" in r and f"{mname}|mse" in r]
                if len(both) < 5:
                    continue
                L = np.array([np.log(r[f"{ref}|mse"] / r[f"{mname}|mse"]) for r in both])   # (n, 7)
                p = holm([wilcoxon_p(L[:, j]) if np.isfinite(wilcoxon_p(L[:, j])) else 1.0 for j in range(len(BUDGETS))])
                gm = np.exp(L.mean(0))
                report[sname][f"{mname}_vs_{ref}"] = {"n": len(both), "geo_ratio": gm.tolist(), "p_holm": p.tolist(),
                                                     "wins": (L > TIE).sum(0).tolist()}
                print(f"{mname + ' vs ' + ref:<28}" + "".join(f"{g:>6.2f}({q:.0e})" for g, q in zip(gm, p)))

    con = subsets["connected"]
    both = [r for r in rows if "qdislib|mse" in r]
    fc_better = np.mean([r["find_cuts|mse"][0] < r["qdislib|mse"][0] - TIE for r in both])
    print(f"\nL1 find_cuts lower MSE than Qdislib at 1e3: {fc_better:.3f} of {len(both)} circuits (prediction >= 0.70)")
    fq = [(r["qdislib|F"], r["random|F"]) for r in both if np.isfinite(r["qdislib|F"])]
    dq = np.array([a - b for a, b in fq])
    print(f"L2 Qdislib F - random F: mean {dq.mean():+.4f}, Wilcoxon p {wilcoxon_p(dq):.2e}, n {len(dq)} "
          f"(prediction: no difference)")
    gain_fc = np.mean([r["oracle_fc"][0] < r["oracle_cand"][0] - TIE for r in rows])
    gain_qd = np.mean([r["oracle_qd"][0] < r["oracle_cand"][0] - TIE for r in rows])
    print(f"L3 adding the cut lowers the candidate oracle at 1e3: find_cuts {gain_fc:.3f}, Qdislib {gain_qd:.3f} "
          f"(prediction: Qdislib < find_cuts; Qdislib top5/mixed cuts were not labelled)")
    print("Union headroom, geometric mean of candidate-oracle MSE / union-oracle MSE:")
    for sname, rs in (("all", rows), ("connected", con)):
        u = np.exp(np.mean([np.log(r["oracle_cand"] / r["oracle"]) for r in rs], axis=0))
        print(f"  {sname:<10}" + "".join(f"{x:>8.3f}" for x in u))

    with open(OUT / "per_circuit.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(np.asarray(v).tolist()) if isinstance(v, np.ndarray) else v) for k, v in r.items()})
    (OUT / "tests.json").write_text(json.dumps(report, indent=1))
    print(f"\nwritten {OUT / 'per_circuit.csv'} and {OUT / 'tests.json'}")


if __name__ == "__main__":
    main()
