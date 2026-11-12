"""Device-transfer test of the frozen models (notes/PREREG_transfer_sherbrooke.md). Run once.

The 15 frozen checkpoints and the fitted prior and linear weights (trained on fake_brisbane labels)
are applied, unchanged, to the holdout4 circuits relabelled on fake_sherbrooke
(paper_budget.datagen.relabel_device). The hardware features and the layout-ESP are computed from
the calibration of the device the labels come from (paper_budget.device.use_device). Nothing is
trained, refitted or tuned. The metrics and tests are those of eval_holdout4.py.

  Reproduction check (before any new label exists; may be run any number of times):
    QUEST_UNBLIND=holdout4 PYTHONHASHSEED=0 python -m paper_budget.eval_transfer --check-holdout4
    -> recomputes eval4_holdout4.json with this code on fake_brisbane; stops on any difference > 1e-6.
  The test (once):
    QUEST_UNBLIND=holdout4_sherbrooke PYTHONHASHSEED=0 python -m paper_budget.eval_transfer \\
        --test holdout4_sherbrooke --device fake_sherbrooke

Families (PREREG_transfer_sherbrooke.md section 4):
  T  (primary)   M2b vs M1 at 1e8 and 1e9, Holm over 2              -> T1 (the H5 criteria)
  S  (secondary) M2b vs M0 and M0 vs min_kappa2 at 1e8 and 1e9, Holm over 4 -> S1, S2
  R  (replication, descriptive) M2 vs M1 and M2 vs M3, Holm over 4
  C  (calibration control, exploratory) M0 and M2b with the labels' device calibration vs the
     fake_brisbane calibration, on the same labels, at 1e8 and 1e9 (no adjustment)
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from paper_budget.analysis.deployable_rule import holm
from paper_budget.config import BUDGETS, DATA_V2
from paper_budget.device import current_device, use_device
from paper_budget.eval_gt_att import H2_MARGIN, PRIMARY, fit_metrics, gm, load_test, log_regret
from paper_budget.eval_holdout4 import RUN, SEEDS, _pred, paired, verify_frozen
from paper_budget.models_att import DirectGT, ResidualGT, clip_att, predict_m0, predict_m1

TRAIN_DEVICE = "fake_brisbane"
MODELS = ("M0", "M1", "M2", "M2b", "M3")
WHERE = {"M2": "main", "M3": "main", "M2b": "m2b"}


def frozen_config() -> tuple:
    cfg = json.loads((RUN / "main" / "config.json").read_text())
    cfg_b = json.loads((RUN / "m2b" / "config.json").read_text())
    assert cfg_b["alpha"] == cfg["alpha"] and cfg_b["beta"] == cfg["beta"] and cfg_b["w"] == cfg["w"]
    return cfg, (cfg["alpha"], cfg["beta"]), np.asarray(cfg["w"]), cfg["a_min"]


def load_nets(dev) -> dict:
    nets = {}
    for m in ("M2", "M2b", "M3"):
        nets[m] = []
        for s in SEEDS:
            ck = torch.load(RUN / WHERE[m] / f"{m}_seed{s}.pt", map_location=dev)
            net = (DirectGT if m == "M3" else ResidualGT)().to(dev)
            net.load_state_dict(ck["state_dict"])
            net.eval()
            nets[m].append(net)
    return nets


def predict(C, nets, ab, w, a_min, dev) -> tuple:
    """Clipped predictions of the five models (Transformers seed-averaged), as eval_holdout4.py."""
    base = {"M2": lambda c: predict_m0(c, ab), "M2b": lambda c: predict_m1(c, ab, w),
            "M3": lambda c: np.zeros(c.n_cand)}
    raw = {"M0": [predict_m0(c, ab) for c in C], "M1": [predict_m1(c, ab, w) for c in C]}
    per_seed = {}
    for m in ("M2", "M2b", "M3"):
        per_seed[m] = [[_pred(net, c, base[m](c), dev) for c in C] for net in nets[m]]
        raw[m] = [np.mean([sp[i] for sp in per_seed[m]], axis=0) for i in range(len(C))]
    preds = {k: [clip_att(q, a_min)[0] for q in P] for k, P in raw.items()}
    return preds, per_seed


def regrets(C, preds) -> dict:
    L = {k: np.array([log_regret(c, q) for c, q in zip(C, P)]) for k, P in preds.items()}
    mo = [int(np.lexsort((c.e, c.g2))[0]) for c in C]
    L["min_kappa2"] = np.array([[np.log((c.b2[k] + c.v[k] / N) / max((c.b2 + c.v / N).min(), 1e-300))
                                 for N in np.asarray(BUDGETS, float)] for c, k in zip(C, mo)])
    return L


def with_holm(tests: list) -> list:
    for t, q in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = float(q)
    return tests


def check_holdout4(nets, ab, w, a_min, dev) -> None:
    assert os.environ.get("QUEST_UNBLIND") == "holdout4", "set QUEST_UNBLIND=holdout4"
    use_device(TRAIN_DEVICE)
    C, _ = load_test("holdout4")
    L = regrets(C, predict(C, nets, ab, w, a_min, dev)[0])
    saved = json.loads((RUN / "eval4_holdout4.json").read_text())
    assert saved["n_circuits"] == len(C), f"{len(C)} circuits now, {saved['n_circuits']} in eval4_holdout4.json"
    worst = 0.0
    for k, v in saved["regret_gm"].items():
        d = float(np.max(np.abs(gm(L[k]) - np.asarray(v))))
        worst = max(worst, d)
        print(f"reproduction of eval4_holdout4.json, {k:<10}: max |difference| {d:.2e}")
    if worst > 1e-6:
        raise RuntimeError("eval_transfer does not reproduce eval4_holdout4.json; stop and investigate")
    print("REPRODUCTION OK: this code gives the confirmatory holdout4 results exactly")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=None, help="relabelled split, e.g. holdout4_sherbrooke")
    ap.add_argument("--device", default=None, help="device of the relabelled split, e.g. fake_sherbrooke")
    ap.add_argument("--check-holdout4", action="store_true")
    a = ap.parse_args()
    verify_frozen()
    cfg, ab, w, a_min = frozen_config()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = load_nets(dev)
    if a.check_holdout4:
        check_holdout4(nets, ab, w, a_min, dev)
        return

    assert a.test and a.device, "give --test <split> and --device <device>"
    assert a.device != TRAIN_DEVICE, "the transfer test needs a device other than the training device"
    assert os.environ.get("QUEST_UNBLIND") == a.test, f"set QUEST_UNBLIND={a.test}"
    assert os.environ.get("PYTHONHASHSEED") == "0", "export PYTHONHASHSEED=0 first"
    man = json.loads((DATA_V2 / f"manifest_{a.test}.json").read_text())
    assert man["device"] == a.device, f"{a.test} was labelled on {man['device']}, not {a.device}"
    out = RUN / f"eval_transfer_{a.test}.json"
    if out.exists():
        raise FileExistsError(f"{out} exists: the transfer test is run once")

    use_device(a.device)                                   # features from the labels' own device
    C, _ = load_test(a.test)
    assert not (set(cfg["train_cids"]) | set(cfg["val_cids"])) & {c.cid for c in C}
    for c in C:
        devs = {r["device"] for r in c.g.records}
        assert devs == {a.device}, f"{c.cid}: records from {devs}"
    print(f"{a.test}: {len(C)} circuits; 15 checkpoints verified; features from {current_device()}")
    preds, per_seed = predict(C, nets, ab, w, a_min, dev)
    L = regrets(C, preds)

    use_device(TRAIN_DEVICE)                               # control C: same labels, wrong calibration
    C_mis, _ = load_test(a.test)
    assert [c.cid for c in C_mis] == [c.cid for c in C], "calibration control changed the analysis set"
    print(f"calibration control: features from {current_device()} on the same {len(C_mis)} circuits")
    L_mis = regrets(C_mis, predict(C_mis, nets, ab, w, a_min, dev)[0])
    use_device(a.device)

    fam_t = with_holm([paired(L, "M2b", "M1", j) for j in PRIMARY])
    fam_s = with_holm([paired(L, "M2b", "M0", j) for j in PRIMARY]
                      + [paired(L, "M0", "min_kappa2", j) for j in PRIMARY])
    fam_r = with_holm([paired(L, "M2", ref, j) for ref in ("M1", "M3") for j in PRIMARY])
    both = {"M0": L["M0"], "M2b": L["M2b"], "M0_brisbane_cal": L_mis["M0"], "M2b_brisbane_cal": L_mis["M2b"]}
    fam_c = [paired(both, m, f"{m}_brisbane_cal", j) for m in ("M0", "M2b") for j in PRIMARY]
    fit = {k: fit_metrics(C, P) for k, P in preds.items()}

    def met(tests, ref, margin=1.0, strict=True):
        sel = [t for t in tests if t["ref"] == ref]
        return all((t["ratio"] < margin if strict else t["ratio"] <= margin) and t["p_holm"] < 0.05 for t in sel)

    res = {"test": a.test, "device": a.device, "train_device": TRAIN_DEVICE, "n_circuits": len(C),
           "n_candidates": int(sum(c.n_cand for c in C)),
           "fit": fit, "regret_gm": {k: gm(v).tolist() for k, v in L.items()},
           "regret_gm_brisbane_cal": {k: gm(v).tolist() for k, v in L_mis.items()},
           "family_T": fam_t, "family_S": fam_s, "family_R": fam_r, "control_C": fam_c,
           "T1_met": met(fam_t, "M1", H2_MARGIN, strict=False),
           "S1_met": met(fam_s, "M0"), "S2_met": met(fam_s, "min_kappa2"),
           "seed_regret_gm": {m: [gm(np.array([log_regret(c, clip_att(q, a_min)[0]) for c, q in zip(C, sp)])).tolist()
                                  for sp in per_seed[m]] for m in per_seed}}
    by = defaultdict(list)
    for i, c in enumerate(C):
        by[f"{c.g.family}"].append(i)
        by[f"n{c.g.n_qubits}"].append(i)
    res["by_group_regret_gm"] = {g: {"n": len(idx), **{k: gm(L[k][idx]).tolist() for k in (*MODELS, "min_kappa2")}}
                                 for g, idx in sorted(by.items())}
    h4 = json.loads((RUN / "eval4_holdout4.json").read_text())
    res["holdout4_brisbane_regret_gm"] = h4["regret_gm"]
    out.write_text(json.dumps(res, indent=1))
    (RUN / f"eval_transfer_{a.test}_per_circuit.json").write_text(json.dumps(
        [{"cid": c.cid, "family": c.g.family, "n_qubits": int(c.g.n_qubits), "n_cand": c.n_cand,
          **{k: L[k][i].tolist() for k in L}, **{f"{k}_brisbane_cal": L_mis[k][i].tolist() for k in ("M0", "M2b")}}
         for i, c in enumerate(C)]))

    print("\nfit (seed-averaged, clipped)")
    for k, v in fit.items():
        print(f"  {k:<4} R2_within {v['r2_within']:+.4f}  tau_b {v['kendall_tau_b']:.4f}  rmse {v['rmse']:.4f}")
    print(f"\ngeometric-mean regret on {a.device} (true m)   [holdout4 on {TRAIN_DEVICE} in brackets]")
    print(f"{'budget':>8}" + "".join(f"{k:>18}" for k in L))
    for j, N in enumerate(BUDGETS):
        print(f"{N:>8.0e}" + "".join(f"{res['regret_gm'][k][j]:>9.4f} [{h4['regret_gm'][k][j]:.4f}]" for k in L))
    for title, tests in (("Family T (primary)", fam_t), ("Family S (secondary)", fam_s),
                         ("Family R (replication, descriptive)", fam_r),
                         ("Control C (exploratory; ratio < 1 = own calibration better)", fam_c)):
        print(f"\n{title}  (ratio = gm regret model / reference; < 1 = model better)")
        for t in tests:
            ph = f"  Holm {t['p_holm']:.2e}" if "p_holm" in t else ""
            print(f"  {t['model']} vs {t['ref']} at {t['budget']:.0e}: ratio {t['ratio']:.4f} "
                  f"CI [{t['ci95'][0]:.4f}, {t['ci95'][1]:.4f}]  p {t['p']:.2e}{ph}  "
                  f"different choices {t['n_differ']}/{len(C)}")
    print(f"\nT1 {res['T1_met']}  S1 {res['S1_met']}  S2 {res['S2_met']}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
