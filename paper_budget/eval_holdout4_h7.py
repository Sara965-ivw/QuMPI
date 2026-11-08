"""Amendment 1 to PREREG_holdout4.md: M2b vs M1 by overhead group (H7, H8). Run once per test set.

  QUEST_UNBLIND=holdout4 python -m paper_budget.eval_holdout4_h7 --test holdout4   (confirmatory for H7)
Does not change eval_holdout4.py, H5 or any model. Circuits are the same analysis set as eval_holdout4.py;
a circuit is "one overhead" when all its candidates share one sampling overhead (log2 kappa^2 equal after
rounding to 1e-9). Family H7 (Holm over 2): M2b vs M1 at 1e8 and 1e9 on one-overhead circuits.
"""
import argparse
import json
import os

import numpy as np
import torch

from paper_budget.analysis.deployable_rule import holm
from paper_budget.config import BUDGETS
from paper_budget.eval_gt_att import PRIMARY, boot_ratio, load_test, log_regret, wilcoxon_p
from paper_budget.eval_holdout4 import RUN, SEEDS, _pred, verify_frozen
from paper_budget.models_att import ResidualGT, clip_att, predict_m1

H7_MARGIN = 0.95
H8_BAND = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    name = ap.parse_args().test
    assert os.environ.get("QUEST_UNBLIND") == name, f"set QUEST_UNBLIND={name}"
    out = RUN / f"eval4_h7_{name}.json"
    if out.exists():
        raise FileExistsError(f"{out} exists: the evaluation is run once")
    verify_frozen()
    cfg = json.loads((RUN / "main" / "config.json").read_text())
    ab, w, a_min = (cfg["alpha"], cfg["beta"]), np.asarray(cfg["w"]), cfg["a_min"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = []
    for s in SEEDS:
        n = ResidualGT().to(dev)
        n.load_state_dict(torch.load(RUN / "m2b" / f"M2b_seed{s}.pt", map_location=dev)["state_dict"])
        n.eval()
        nets.append(n)
    C, _ = load_test(name)
    L = {"one": {"M1": [], "M2b": []}, "several": {"M1": [], "M2b": []}}
    for c in C:
        base = predict_m1(c, ab, w)
        g = "one" if len(set(np.round(np.log2(c.g2), 9))) == 1 else "several"
        L[g]["M1"].append(log_regret(c, clip_att(base, a_min)[0]))
        L[g]["M2b"].append(log_regret(c, clip_att(np.mean([_pred(n, c, base, dev) for n in nets], axis=0),
                                                  a_min)[0]))
    res = {"test": name, "n_circuits": len(C)}
    for g in L:
        a, b = np.array(L[g]["M2b"]), np.array(L[g]["M1"])
        res[g] = {"n": len(a)}
        if len(a) >= 5:
            res[g]["ratio"] = np.exp((a - b).mean(axis=0)).tolist()
            res[g]["ci95"] = boot_ratio(a, b).tolist()
            res[g]["p"] = [wilcoxon_p((b - a)[:, j]) for j in range(len(BUDGETS))]
            res[g]["n_differ"] = [int(np.sum(np.abs(a - b)[:, j] > 1e-12)) for j in range(len(BUDGETS))]
    one = res["one"]
    if "ratio" in one:
        ph = holm([one["p"][j] for j in PRIMARY])
        one["p_holm_primary"] = ph.tolist()
        res["H7_met"] = all(one["ratio"][j] <= H7_MARGIN and q < 0.05 for j, q in zip(PRIMARY, ph))
    else:
        res["H7_met"] = None                                  # fewer than 5 one-overhead circuits
    sev = res["several"]
    res["H8_met"] = (all(abs(sev["ratio"][j] - 1) < H8_BAND for j in PRIMARY) if "ratio" in sev else None)
    out.write_text(json.dumps(res, indent=1))
    print(f"{name}: {len(C)} circuits; one overhead {one['n']} ({one['n'] / max(len(C), 1):.1%}), "
          f"several {sev['n']}; checkpoints verified")
    for g in ("one", "several"):
        r = res[g]
        if "ratio" not in r:
            print(f"  {g}: too few circuits")
            continue
        for j in PRIMARY:
            print(f"  {g:<8} {BUDGETS[j]:.0e}: M2b/M1 {r['ratio'][j]:.4f} CI [{r['ci95'][0][j]:.4f}, "
                  f"{r['ci95'][1][j]:.4f}]  p {r['p'][j]:.2e}  different choices {r['n_differ'][j]}/{r['n']}")
    print(f"H7 {res['H7_met']}  H8 {res['H8_met']}\nwritten {out}")


if __name__ == "__main__":
    main()
