"""Frozen budget-aware models vs METIS, random, IBM find_cuts and Qdislib on holdout4 (EXPLORATORY).

Status: exploratory, as listed in notes/PREREG_holdout4.md section 8 (no confirmatory claim).
Predictions P1-P4 were committed before any find_cuts or Qdislib result on holdout4 existed:
notes/PREDICTIONS_compare_holdout4.md. The models are the 15 frozen checkpoints (SHA-256 verified);
nothing is trained or refitted.

Circuits: the 156 of the holdout4 analysis set (data/corpus_v2/holdout4_analysis_set.json). A
circuit whose find_cuts labelling timed out or failed is removed from every method and listed (the
October rule). A circuit without a usable Qdislib result is removed from the Qdislib tests only.

Every cut is scored with the same exact finite-shot error (paper_budget.targets.bias_variance):
    MSE_k(N) = b2_k + v_k / N,   N = 1e3 ... 1e9 shots.
Methods (one cut per circuit and budget):
  M2b, M1, M0   budget-aware rule with the true m, exactly as eval_holdout4.py (seed-averaged, clipped)
  min_kappa2    lowest sampling overhead gamma^2, ties broken by ESP (as eval_holdout4.py)
  metis         METIS-closest candidate (quest_phase2.baselines.score_metis, the RQ1 baseline)
  random        expectation over a uniform pick among the candidates
  find_cuts     the labelled qiskit-addon-cutting find_cuts cut (may lie outside the candidates,
                including the no-cut option of a separable circuit)
  qdislib       Qdislib default find_cut, 5 seeds averaged; an unlabelled seed gets the optimistic
                bound (gamma^2 - 1) / N (decision Q3, notes/DECISIONS_qdislib.md)
Oracle: the best of the candidates, the find_cuts cut and the labelled Qdislib cuts, per budget.
Separable: find_cuts needs no cut (label kind "nocut").

Statistics, per comparison: L = log(MSE_reference / MSE_method) per circuit; geometric-mean ratio
exp(mean L) (> 1: method better); Wilcoxon signed-rank, Holm over the 7 budgets (as compare_tools.py);
95% bootstrap CI over circuits (10 000 resamples, seed 2026); wins / ties / losses.

  QUEST_UNBLIND=holdout4 PYTHONHASHSEED=0 python -m paper_budget.analysis.compare_holdout4
Run once: the script refuses to run if runs/paper_budget/compare_holdout4/tests.json exists.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import paper_budget.att_data  # noqa: F401  (puts stage2_predictor on sys.path)
from paper_budget.analysis.compare_tools import TIE, holm, label_mse, qdislib_cut, wilcoxon_p
from paper_budget.analysis.deployable_rule import choose
from paper_budget.config import BUDGETS, DATA_V2, ROOT, RUNS

N = np.asarray(BUDGETS, float)
FC_LAB = ROOT / "runs" / "findcuts_label_holdout4"
QD = RUNS / "qdislib_holdout4"
OUT = RUNS / "compare_holdout4"
SELECTION = DATA_V2 / "holdout4_analysis_set.json"
EVAL = RUNS / "gt_att" / "eval4_holdout4.json"
N_BOOT, BOOT_SEED = 10_000, 2026
OURS = ("M2b", "M1", "M0", "min_kappa2")
REFS = ("find_cuts", "qdislib", "metis", "random")
METHODS = OURS + ("metis", "random", "find_cuts", "qdislib")
HI, LO = (5, 6), 0                      # indices of 1e8, 1e9 and 1e3 in BUDGETS
P1_MARGIN = 1.10                        # PREDICTIONS_compare_holdout4.md, P1


# ---------------------------------------------------------------- pure functions (tested)
def candidate_mse(b2: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(K,) bias^2 and per-shot variance -> (K, 7) exact MSE of every candidate at every budget."""
    return np.asarray(b2, float)[:, None] + np.asarray(v, float)[:, None] / N[None, :]


def budget_rule_mse(f_hat: np.ndarray, g2: np.ndarray, m: float, M: np.ndarray) -> np.ndarray:
    """(7,) true MSE of the cut the budget-aware rule picks at each budget from predicted f_hat."""
    return np.array([M[choose(f_hat, g2, m, Nj), j] for j, Nj in enumerate(N)])


def paired_stats(L: np.ndarray, seed: int = BOOT_SEED) -> dict:
    """L (n, 7) = log(MSE_ref / MSE_method). Geometric ratio, bootstrap CI, Holm-adjusted Wilcoxon."""
    L = np.asarray(L, float)
    raw = [wilcoxon_p(L[:, j]) for j in range(L.shape[1])]
    p = holm([q if np.isfinite(q) else 1.0 for q in raw])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(L), size=(N_BOOT, len(L)))
    ci = np.quantile(np.exp(L[idx].mean(axis=1)), [0.025, 0.975], axis=0)
    return {"n": int(len(L)), "geo_ratio": np.exp(L.mean(0)).tolist(), "ci95": ci.tolist(),
            "p_holm": p.tolist(), "wins": (L > TIE).sum(0).tolist(), "ties": (np.abs(L) <= TIE).sum(0).tolist(),
            "losses": (L < -TIE).sum(0).tolist()}


def gm_vs_oracle(rows: list[dict], name: str) -> tuple[np.ndarray, int]:
    """Geometric mean over circuits of MSE(method) / MSE(oracle), per budget, and n used."""
    have = [r for r in rows if name in r]
    if not have:
        return np.full(len(N), np.nan), 0
    return np.exp(np.mean([np.log(r[name] / r["oracle"]) for r in have], axis=0)), len(have)


def compare(rows: list[dict], method: str, ref: str) -> dict | None:
    both = [r for r in rows if method in r and ref in r]
    if len(both) < 5:
        return None
    return paired_stats(np.array([np.log(r[ref] / r[method]) for r in both]))


def judge(report: dict) -> dict:
    """Outcome of P1-P4 and D1 exactly as worded in PREDICTIONS_compare_holdout4.md."""
    def ok(t, j, better):                          # better: method lower MSE (ratio > 1)
        if t is None:
            return None
        r, p = t["geo_ratio"][j], t["p_holm"][j]
        return bool(p < 0.05 and (r > 1 if better else r < 1))

    con, al = report["connected"], report["all"]
    fc_c, qd_c = con.get("M2b_vs_find_cuts"), con.get("M2b_vs_qdislib")
    out = {
        "P1": None if fc_c is None else all(fc_c["geo_ratio"][j] >= P1_MARGIN and fc_c["p_holm"][j] < 0.05
                                            for j in HI),
        "P2": None if qd_c is None else all(ok(qd_c, j, True) for j in HI),
        "P3": None if al.get("M2b_vs_find_cuts") is None or al.get("M2b_vs_qdislib") is None else
              ok(al["M2b_vs_find_cuts"], LO, False) and ok(al["M2b_vs_qdislib"], LO, False),
        "P4": None if al.get("M2b_vs_metis") is None else
              all(ok(al["M2b_vs_metis"], j, True) and ok(al["M2b_vs_random"], j, True) for j in HI),
    }
    cross = [j for j in range(len(N)) if fc_c is not None and ok(fc_c, j, True)]
    out["D1_first_budget_M2b_beats_find_cuts_connected"] = float(N[cross[0]]) if cross else None
    return out


# ---------------------------------------------------------------- data and models
def load_models():
    import torch

    from paper_budget.eval_holdout4 import RUN, SEEDS, verify_frozen
    from paper_budget.models_att import ResidualGT
    verify_frozen()
    cfg = json.loads((RUN / "main" / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = []
    for s in SEEDS:
        net = ResidualGT().to(device)
        net.load_state_dict(torch.load(RUN / "m2b" / f"M2b_seed{s}.pt", map_location=device)["state_dict"])
        net.eval()
        nets.append(net)
    return cfg, nets, device


def model_predictions(c, cfg, nets, device) -> dict:
    """Clipped attenuation predictions of M0, M1 and the seed-averaged M2b (as eval_holdout4.py)."""
    from paper_budget.eval_holdout4 import _pred
    from paper_budget.models_att import clip_att, predict_m0, predict_m1
    ab, w, a_min = (cfg["alpha"], cfg["beta"]), np.asarray(cfg["w"]), cfg["a_min"]
    base = predict_m1(c, ab, w)
    return {"M0": clip_att(predict_m0(c, ab), a_min)[0], "M1": clip_att(base, a_min)[0],
            "M2b": clip_att(np.mean([_pred(n, c, base, device) for n in nets], axis=0), a_min)[0]}


def check_reproduction(C, preds) -> None:
    """Regret against the best candidate must equal eval4_holdout4.json (same models, same circuits)."""
    saved = json.loads(EVAL.read_text())["regret_gm"]
    for k in ("M0", "M1", "M2b"):
        g = []
        for c, p in zip(C, preds):
            M = candidate_mse(c.b2, c.v)
            g.append(np.log(budget_rule_mse(np.exp(-p[k]), c.g2, c.m, M) / np.maximum(M.min(0), 1e-300)))
        diff = np.max(np.abs(np.exp(np.mean(g, axis=0)) - np.asarray(saved[k])))
        print(f"reproduction of eval4_holdout4.json, {k}: max |difference| {diff:.2e}")
        if diff > 1e-6:
            raise RuntimeError(f"{k} does not reproduce eval4_holdout4.json; stop and investigate")


def qdislib_entry(cid: str, ideal) -> tuple[np.ndarray | None, list, str | None, bool]:
    """(seed-averaged MSE or None, labelled Qdislib cut MSEs for the oracle, reason if missing, bound used)."""
    sel, lab = QD / "select", QD / "labels"
    if not (sel / f"{cid}.json").exists() or not (lab / f"{cid}.json").exists():
        return None, [], "not run", False
    cuts = [label_mse(x["label"], ideal) for x in json.loads((lab / f"{cid}.json").read_text())["cuts"]
            if x["label"]]
    qd, why = qdislib_cut(cid, ideal, sel, lab)
    if qd is None:
        return None, cuts, why, False
    return qd[0], cuts, None, bool(qd[2])


# ---------------------------------------------------------------- main
def main():
    from paper_budget.eval_gt_att import load_test
    from quest_phase2.baselines import metis_method_used, score_metis  # noqa: E402  (frozen RQ1 baseline)

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-qdislib", action="store_true",
                    help="run without Qdislib; P2 and P3 are then reported as not evaluated")
    a = ap.parse_args()
    assert os.environ.get("QUEST_UNBLIND") == "holdout4", "set QUEST_UNBLIND=holdout4"
    assert os.environ.get("PYTHONHASHSEED") == "0", "export PYTHONHASHSEED=0 first"
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "tests.json").exists():
        raise FileExistsError(f"{OUT / 'tests.json'} exists: the comparison is run once")

    cfg, nets, device = load_models()
    C, _ = load_test("holdout4")
    listed = json.loads(SELECTION.read_text())["circuits"]
    assert sorted(c.cid for c in C) == sorted(listed), "analysis set differs from holdout4_analysis_set.json"
    metis_method = metis_method_used([C[0].g])
    print(f"holdout4: {len(C)} circuits; 15 checkpoints verified; device {device}; "
          f"METIS partitioner: {metis_method}")
    preds = [model_predictions(c, cfg, nets, device) for c in C]
    check_reproduction(C, preds)

    rows, excluded = [], []
    for c, p in zip(C, preds):
        d = json.loads((DATA_V2 / "holdout4" / f"{c.cid}.json").read_text())
        ideal = d["obs_ideal"]
        assert len(d["records"]) == c.n_cand
        fl = json.loads((FC_LAB / f"{c.cid}.json").read_text())
        if fl["status"] != "ok" or fl["label"] is None:
            excluded.append((c.cid, fl["status"]))
            continue
        M = candidate_mse(c.b2, c.v)
        row = {"cid": c.cid, "family": c.g.family, "n_qubits": int(c.g.n_qubits),
               "separable": fl["kind"] == "nocut", "fc_kind": fl["kind"], "fc_width": fl["width_used"],
               "fc_reused": fl["reused_candidate_label"]}
        for k in ("M2b", "M1", "M0"):
            row[k] = budget_rule_mse(np.exp(-p[k]), c.g2, c.m, M)
        row["min_kappa2"] = M[int(np.lexsort((c.e, c.g2))[0])]
        row["metis"] = M[int(np.argmax(score_metis(c.g)))]
        row["random"] = M.mean(0)
        row["find_cuts"] = label_mse(fl["label"], ideal)
        row["oracle_cand"] = M.min(0)
        oracle = np.minimum(row["oracle_cand"], row["find_cuts"])
        if not a.no_qdislib:
            qd, qd_cuts, why, bound = qdislib_entry(c.cid, ideal)
            if qd is None:
                row["qdislib_missing"] = why
            else:
                row["qdislib"], row["qdislib_bound"] = qd, bound
            if qd_cuts:
                oracle = np.minimum(oracle, np.min(qd_cuts, axis=0))
        row["oracle"] = oracle
        rows.append(row)

    subsets = {"all": rows, "connected": [r for r in rows if not r["separable"]],
               "separable": [r for r in rows if r["separable"]]}
    hdr = "".join(f"{'1e' + str(int(np.log10(b))):>9}" for b in BUDGETS)
    print(f"compared: {len(rows)}; excluded (find_cuts not labelled): {excluded}")
    print(f"Qdislib missing: {[(r['cid'], r['qdislib_missing']) for r in rows if 'qdislib_missing' in r]}")
    print(f"Qdislib with the optimistic bound on some seed: {sum(bool(r.get('qdislib_bound')) for r in rows)}")
    report = {"n_compared": len(rows), "excluded": excluded, "metis_method": metis_method,
              "n_by_subset": {k: len(v) for k, v in subsets.items()}, "gm_vs_oracle": {}, "headroom": {}}
    for sname, rs in subsets.items():
        print(f"\n== {sname} (n = {len(rs)}): geometric mean of MSE / oracle")
        print(f"{'method':<12}{hdr}")
        report["gm_vs_oracle"][sname] = {}
        for mname in METHODS:
            g, n = gm_vs_oracle(rs, mname)
            report["gm_vs_oracle"][sname][mname] = {"n": n, "gm": g.tolist()}
            print(f"{mname:<12}" + "".join(f"{x:>9.3f}" for x in g) + ("" if n == len(rs) else f"  (n={n})"))
        h, _ = gm_vs_oracle(rs, "oracle_cand")
        report["headroom"][sname] = h.tolist()
        print(f"{'candidates':<12}" + "".join(f"{x:>9.3f}" for x in h) + "  (best candidate / union oracle)")
        print("-- paired: geometric ratio reference / method (> 1 = method better), Holm p over 7 budgets")
        report[sname] = {}
        for mname in OURS:
            for ref in REFS:
                t = compare(rs, mname, ref)
                if t is None:
                    continue
                report[sname][f"{mname}_vs_{ref}"] = t
                print(f"{mname + ' vs ' + ref:<24}" + "".join(
                    f"{g:>6.2f}({q:.0e})" for g, q in zip(t["geo_ratio"], t["p_holm"])))

    verdict = judge(report)
    report["predictions"] = verdict
    print("\nPredictions (notes/PREDICTIONS_compare_holdout4.md):")
    for k, v in verdict.items():
        print(f"  {k}: {'not evaluated' if v is None else v}")
    for sname in ("all", "connected"):
        for ref in ("find_cuts", "qdislib"):
            t = report[sname].get(f"M2b_vs_{ref}")
            if t:
                for j in HI:
                    print(f"  {sname}, M2b vs {ref} at {N[j]:.0e}: ratio {t['geo_ratio'][j]:.3f} "
                          f"CI [{t['ci95'][0][j]:.3f}, {t['ci95'][1][j]:.3f}]  Holm p {t['p_holm'][j]:.1e}  "
                          f"W/T/L {t['wins'][j]}/{t['ties'][j]}/{t['losses'][j]}")
    (OUT / "tests.json").write_text(json.dumps(report, indent=1))
    with open(OUT / "per_circuit.json", "w") as fh:
        json.dump([{k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                   for r in rows], fh)
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
