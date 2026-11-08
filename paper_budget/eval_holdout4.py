"""Confirmatory evaluation on holdout4 (notes/PREREG_holdout4.md). Run once per test set.

  QUEST_UNBLIND=holdout4 python -m paper_budget.eval_holdout4 --test holdout4      (confirmatory)
  QUEST_UNBLIND=holdout3 python -m paper_budget.eval_holdout4 --test holdout3      (exploratory)
  python -m paper_budget.eval_holdout4 --val-check                                 (pipeline check)
Before anything is read, the SHA-256 of all 15 frozen checkpoints is compared with the values in the
pre-registration (FROZEN below); any mismatch stops the evaluation. Nothing is refitted.
Family A (primary, new): M2b vs M1 at 1e8 and 1e9, Holm over 2.
Family B (replication of PREREG_gt_att.md): M2 vs M1 and M2 vs M3 at 1e8 and 1e9, Holm over 4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import torch

from paper_budget.analysis.deployable_rule import holm
from paper_budget.config import BUDGETS, RUNS
from paper_budget.eval_gt_att import (H2_MARGIN, H4_MARGIN, PRIMARY, boot_ratio, fit_metrics, gm,
                                      load_test, log_regret, wilcoxon_p)
from paper_budget.models_att import DirectGT, ResidualGT, clip_att, predict_m0, predict_m1
from quest_phase2.models import to_batch  # noqa: E402  (frozen)

RUN = RUNS / "gt_att"
FROZEN = {
    "main/M2_seed0.pt": "963a208c747a799babe9b52e6897cf78e8e8cd0a3d2334c8eb12792a5e8c4d61",
    "main/M2_seed1.pt": "29b64eda3ccb7b723a67804cc8a7ae4a21304d463dba6e53f48be3c53beb116f",
    "main/M2_seed2.pt": "e2a3e697adfa8e23f668d642261025cd8157dcac8356ae1e04b1f6777f2b6e3c",
    "main/M2_seed3.pt": "30d1d62cfac59e4c1e6ae9e46239347e0cdd0975bc105c1ef2dfbe3d5238c1c8",
    "main/M2_seed4.pt": "5c055be1a59640ea1e136a6a76b15c3b6231a041a7c9ed389b94ffb104d29309",
    "main/M3_seed0.pt": "c28fccd0f00eafd298bf18328525a62fd2b6a1eae9df0b1b41230a661e6531a3",
    "main/M3_seed1.pt": "e335cf1784c170553648c578fa954cf50f1b3bb6409f75e8c765820910859031",
    "main/M3_seed2.pt": "9fd55c0c3526e829427b5d0733e4da06ab5bdcb1b26912f646300608d82ff66d",
    "main/M3_seed3.pt": "4a401fedd5c66ac29e6e38402a49fdfc31c1ba2c712a4df1f9a8a14471d6a486",
    "main/M3_seed4.pt": "a8f0241339459a4fb2cd51335b299a8e8dfaa6ecad515c21aa3e3be6afd7d5ba",
    "m2b/M2b_seed0.pt": "d4229b2846d2eef400214dff5e5c4f2c0a39d1baa37aaec1d170a4f62ab761af",
    "m2b/M2b_seed1.pt": "ba7e4d84d6a7deac26d7a54cd5cd61c3e3cc8add67218fc378af91e00e34ea03",
    "m2b/M2b_seed2.pt": "caf23371276b20a97bb00fffec7f020a2088062eaa301efabdf7693252b3f38a",
    "m2b/M2b_seed3.pt": "96723d3f1263ffd2c5dc7a4e9fb8cf5540e0ebdacff3604b17757e0b135667ec",
    "m2b/M2b_seed4.pt": "ece0c55833a2fcb0426760383a271640210b44f9d9ee86037159527e089f7eb9",
}
SEEDS = (0, 1, 2, 3, 4)


def verify_frozen() -> None:
    for rel, h in FROZEN.items():
        got = hashlib.sha256(open(RUN / rel, "rb").read()).hexdigest()
        if got != h:
            raise RuntimeError(f"{rel}: SHA-256 differs from the pre-registration")


def _pred(model, c, base, device) -> np.ndarray:
    with torch.no_grad():
        p = torch.as_tensor(base, dtype=torch.float32, device=device)
        return model(to_batch(c.g, device), p).cpu().numpy().astype(float)


def paired(L, mod, ref, j):
    d = L[ref][:, j] - L[mod][:, j]                      # > 0: mod better
    return {"model": mod, "ref": ref, "budget": BUDGETS[j], "ratio": float(np.exp(-d.mean())),
            "ci95": boot_ratio(L[mod], L[ref])[:, j].tolist(), "p": wilcoxon_p(d),
            "n_differ": int(np.sum(np.abs(d) > 1e-12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=None)
    ap.add_argument("--val-check", action="store_true")
    a = ap.parse_args()
    assert a.val_check != bool(a.test), "give --test <name> or --val-check"
    name = "valcheck" if a.val_check else a.test
    if name in ("holdout3", "holdout4"):
        assert os.environ.get("QUEST_UNBLIND") == name, f"set QUEST_UNBLIND={name}"
    out = RUN / f"eval4_{name}.json"
    if out.exists():
        raise FileExistsError(f"{out} exists: the evaluation is run once")
    verify_frozen()
    cfg = json.loads((RUN / "main" / "config.json").read_text())
    cfg_b = json.loads((RUN / "m2b" / "config.json").read_text())
    assert cfg_b["alpha"] == cfg["alpha"] and cfg_b["beta"] == cfg["beta"] and cfg_b["w"] == cfg["w"]
    ab, w, a_min = (cfg["alpha"], cfg["beta"]), np.asarray(cfg["w"]), cfg["a_min"]
    if a.val_check:
        C, _ = load_test(cfg["train_corpus"], set(cfg["val_cids"]))
    else:
        C, _ = load_test(name)
        assert not (set(cfg["train_cids"]) | set(cfg["val_cids"])) & {c.cid for c in C}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{name}: {len(C)} circuits; checkpoints verified (15 SHA-256)")

    base = {"M2": lambda c: predict_m0(c, ab), "M2b": lambda c: predict_m1(c, ab, w),
            "M3": lambda c: np.zeros(c.n_cand)}
    where = {"M2": "main", "M3": "main", "M2b": "m2b"}
    raw = {"M0": [predict_m0(c, ab) for c in C], "M1": [predict_m1(c, ab, w) for c in C]}
    per_seed = defaultdict(list)
    for m in ("M2", "M2b", "M3"):
        for s in SEEDS:
            ck = torch.load(RUN / where[m] / f"{m}_seed{s}.pt", map_location=device)
            net = (DirectGT if m == "M3" else ResidualGT)().to(device)
            net.load_state_dict(ck["state_dict"])
            net.eval()
            per_seed[m].append([_pred(net, c, base[m](c), device) for c in C])
        raw[m] = [np.mean([sp[i] for sp in per_seed[m]], axis=0) for i in range(len(C))]
    preds, n_clip = {}, {}
    for k, P in raw.items():
        cl = [clip_att(q, a_min) for q in P]
        preds[k], n_clip[k] = [q for q, _ in cl], int(sum(n for _, n in cl))

    L = {k: np.array([log_regret(c, q) for c, q in zip(C, P)]) for k, P in preds.items()}
    mo = [int(np.lexsort((c.e, c.g2))[0]) for c in C]
    L["min_kappa2"] = np.array([[np.log((c.b2[k] + c.v[k] / N) / max((c.b2 + c.v / N).min(), 1e-300))
                                 for N in np.asarray(BUDGETS, float)] for c, k in zip(C, mo)])
    fam_a = [paired(L, "M2b", "M1", j) for j in PRIMARY]
    for t, q in zip(fam_a, holm([t["p"] for t in fam_a])):
        t["p_holm"] = float(q)
    fam_b = [paired(L, "M2", ref, j) for ref in ("M1", "M3") for j in PRIMARY]
    for t, q in zip(fam_b, holm([t["p"] for t in fam_b])):
        t["p_holm"] = float(q)
    fit = {k: fit_metrics(C, P) for k, P in preds.items()}
    res = {"test": name, "n_circuits": len(C), "clipped": n_clip, "fit": fit,
           "regret_gm": {k: gm(v).tolist() for k, v in L.items()},
           "family_A": fam_a, "family_B": fam_b,
           "H5_met": all(t["ratio"] <= H2_MARGIN and t["p_holm"] < 0.05 for t in fam_a),
           "H6_met": fit["M2b"]["r2_within"] > fit["M1"]["r2_within"],
           "H2_met": all(t["ratio"] <= H2_MARGIN and t["p_holm"] < 0.05 for t in fam_b if t["ref"] == "M1"),
           "H3_met": all(t["ratio"] < 1 and t["p_holm"] < 0.05 for t in fam_b if t["ref"] == "M3"),
           "H4_met": bool(np.all(gm(L["M2"]) / gm(L["M0"]) <= H4_MARGIN)),
           "H1_met": fit["M2"]["r2_within"] > fit["M1"]["r2_within"],
           "secondary": [paired(L, "M2b", r, j) for r in ("M2", "min_kappa2", "M0") for j in PRIMARY],
           "seed_regret_gm": {m: [gm(np.array([log_regret(c, clip_att(q, a_min)[0]) for c, q in zip(C, sp)])).tolist()
                                  for sp in per_seed[m]] for m in per_seed}}
    fam = defaultdict(list)
    for i, c in enumerate(C):
        fam[c.g.family].append(i)
    res["per_family_regret_gm"] = {f: {k: gm(L[k][idx]).tolist() for k in ("M0", "M1", "M2", "M2b", "M3")}
                                   for f, idx in fam.items()}
    out.write_text(json.dumps(res, indent=1))

    print("\nfit (seed-averaged, clipped):")
    for k, v in fit.items():
        print(f"  {k:<4} R2_within {v['r2_within']:+.4f}  tau_b {v['kendall_tau_b']:.4f}  "
              f"rmse {v['rmse']:.4f}  clipped {n_clip[k]}")
    print("\ngeometric-mean regret (true m)")
    print(f"{'budget':>8}" + "".join(f"{k:>12}" for k in L))
    for j, N in enumerate(BUDGETS):
        print(f"{N:>8.0e}" + "".join(f"{res['regret_gm'][k][j]:>12.4f}" for k in L))
    for title, fam_t in (("Family A (primary): M2b vs M1", fam_a), ("Family B (replication)", fam_b)):
        print(f"\n{title}  (ratio = gm regret model / reference; < 1 = model better)")
        for t in fam_t:
            print(f"  {t['model']} vs {t['ref']} at {t['budget']:.0e}: ratio {t['ratio']:.4f} "
                  f"CI [{t['ci95'][0]:.4f}, {t['ci95'][1]:.4f}]  p {t['p']:.2e}  Holm {t['p_holm']:.2e}  "
                  f"different choices {t['n_differ']}/{len(C)}")
    print(f"\nH5 {res['H5_met']}  H6 {res['H6_met']}  |  H1 {res['H1_met']}  H2 {res['H2_met']}  "
          f"H3 {res['H3_met']}  H4 {res['H4_met']}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
