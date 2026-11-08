"""Frozen M2b vs IBM find_cuts and Qdislib (EXPLORATORY; holdout3; PREREG_holdout4.md section 8).

Same circuits, labels, oracle and tests as compare_tools.py (the October comparison set; find_cuts
labels; Qdislib default find_cut averaged over 5 seeds, optimistic bound for unlabelled seeds), with the
frozen budget-aware models added: M2b and M1 choose a cut per budget with the true m. Also shown: the
minimum-kappa2 rule and the original fidelity-trained Transformer (October picks, budget-blind).
All 15 checkpoints are verified against PREREG_holdout4.md first. Predictions:
notes/PREDICTIONS_compare_m2b.md (committed before the run).

  QUEST_UNBLIND=holdout3 python -m paper_budget.analysis.compare_tools_m2b
"""
import csv
import json
import os

import numpy as np
import torch

from paper_budget.analysis.compare_tools import (FC_CSV, FC_LAB, QD_LAB, N, TIE, holm, label_mse, mse_of,
                                                  qdislib_cut, wilcoxon_p)
from paper_budget.analysis.deployable_rule import choose
from paper_budget.att_data import load_circuit
from paper_budget.config import BUDGETS, DATA_V2, RUNS
from paper_budget.eval_holdout4 import RUN, SEEDS, _pred, verify_frozen
from paper_budget.models_att import ResidualGT, clip_att, predict_m1
from paper_budget.targets import bias_variance

OUT = RUNS / "compare_m2b"


def budget_picks(f, c, M):
    """MSE vector of the cut chosen at each budget by the budget-aware rule with prediction f."""
    return np.array([M[choose(f, c.g2, c.m, Nj), j] for j, Nj in enumerate(N)])


def main():
    assert os.environ.get("QUEST_UNBLIND") == "holdout3", "set QUEST_UNBLIND=holdout3"
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "tests.json").exists():
        raise FileExistsError(f"{OUT / 'tests.json'} exists: run once")
    verify_frozen()
    cfg = json.loads((RUN / "main" / "config.json").read_text())
    ab, w, a_min = (cfg["alpha"], cfg["beta"]), np.asarray(cfg["w"]), cfg["a_min"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = []
    for s in SEEDS:
        net = ResidualGT().to(device)
        net.load_state_dict(torch.load(RUN / "m2b" / f"M2b_seed{s}.pt", map_location=device)["state_dict"])
        net.eval()
        nets.append(net)

    rows, skipped = [], []
    for r in csv.DictReader(open(FC_CSV)):
        cid = r["cid"]
        if r.get("excluded"):
            skipped.append(cid)
            continue
        path = DATA_V2 / "holdout3" / f"{cid}.json"
        d = json.loads(path.read_text())
        ideal = d["obs_ideal"]
        assert int(r["n_cand"]) == len(d["records"]), f"{cid}: candidate count differs from October"
        M = mse_of(bias_variance(d))
        c = load_circuit(path, "holdout3")
        assert c is not None and c.n_cand == len(d["records"])
        m1 = clip_att(predict_m1(c, ab, w), a_min)[0]
        base = predict_m1(c, ab, w)
        m2b = clip_att(np.mean([_pred(net, c, base, device) for net in nets], axis=0), a_min)[0]
        row = {"cid": cid, "separable": r["separable"] == "True",
               "M2b": budget_picks(np.exp(-m2b), c, M), "M1": budget_picks(np.exp(-m1), c, M),
               "min_kappa2": M[int(np.lexsort((c.e, c.g2))[0])],
               "GT_fidelity": M[json.loads(r["model_picks"])].mean(0)}
        fl = json.loads((FC_LAB / f"{cid}.json").read_text())
        row["find_cuts"] = label_mse(fl["label"], ideal)
        qd, why = qdislib_cut(cid, ideal)
        if qd is not None:
            row["qdislib"] = qd[0]
        qd_cuts = [label_mse(x["label"], ideal) for x in
                   json.loads((QD_LAB / f"{cid}.json").read_text())["cuts"] if x["label"]]
        oracle = np.minimum(M.min(0), row["find_cuts"])
        if qd_cuts:
            oracle = np.minimum(oracle, np.min(qd_cuts, axis=0))
        row["oracle"] = oracle
        rows.append(row)

    methods = ["M2b", "M1", "min_kappa2", "GT_fidelity", "find_cuts", "qdislib"]
    subsets = {"all": rows, "connected": [x for x in rows if not x["separable"]]}
    hdr = "".join(f"{'1e' + str(int(np.log10(b))):>9}" for b in BUDGETS)
    print(f"circuits: {len(rows)} (skipped as in October: {len(skipped)}); "
          f"Qdislib available for {sum('qdislib' in x for x in rows)}")
    report = {}
    for sname, rs in subsets.items():
        print(f"\n== {sname} (n = {len(rs)}): geometric mean of MSE / oracle")
        print(f"{'method':<13}{hdr}")
        for mname in methods:
            have = [x for x in rs if mname in x]
            g = np.exp(np.mean([np.log(x[mname] / x["oracle"]) for x in have], axis=0))
            print(f"{mname:<13}" + "".join(f"{v:>9.3f}" for v in g) + ("" if len(have) == len(rs) else f"  (n={len(have)})"))
        print("-- paired: geometric-mean ratio reference / method (> 1 = method better), Holm over 7 budgets")
        report[sname] = {}
        for ref in ("find_cuts", "qdislib"):
            for mname in ("M2b", "M1", "min_kappa2"):
                both = [x for x in rs if ref in x and mname in x]
                L = np.array([np.log(x[ref] / x[mname]) for x in both])
                p = holm([wilcoxon_p(L[:, j]) if np.isfinite(wilcoxon_p(L[:, j])) else 1.0
                          for j in range(len(BUDGETS))])
                gm = np.exp(L.mean(0))
                report[sname][f"{mname}_vs_{ref}"] = {"n": len(both), "geo_ratio": gm.tolist(),
                                                     "p_holm": p.tolist(), "wins": (L > TIE).sum(0).tolist()}
                print(f"{mname + ' vs ' + ref:<24}" + "".join(f"{g:>6.2f}({q:.0e})" for g, q in zip(gm, p)))
    (OUT / "tests.json").write_text(json.dumps(report, indent=1))
    with open(OUT / "per_circuit.json", "w") as fh:
        json.dump([{k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in x.items()}
                   for x in rows], fh)
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
