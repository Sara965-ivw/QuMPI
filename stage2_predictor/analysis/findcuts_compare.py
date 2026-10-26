"""Card A, step 3: compare the cut chosen by each method with the cut chosen by
qiskit-addon-cutting find_cuts, on the 143 holdout3 circuits (decisions A1-A6,
22-23 Oct 2026). Exploratory: holdout3 was opened once on 16 Oct and this
comparison was added afterwards (A1).

Methods (each picks ONE cut per circuit)
  model         frozen primary gt_size-layout, 5 seeds (weights verified by SHA-256
                against partB_record.json); top-scored candidate per seed
  gbm-layout    as in run_rq1_h3 (5 seeds, trained on the same training split)
  esp_layout    highest layout-ESP candidate
  min_overhead  lowest sampling-overhead candidate (the labelled-candidate rule of RQ1)
  metis         METIS-closest candidate (as in RQ1)
  random        expectation over a uniform pick among the candidates
  find_cuts     the cut from findcuts_label.py (may lie outside the candidate set,
                including the no-cut option of a separable circuit)
  oracle        best of the candidates and the find_cuts cut, per metric

Metrics (per circuit)
  F       Hellinger fidelity of the chosen cut (RQ1 label, scheme v3.1)
  regret  F(oracle) - F(chosen)
  MSE(N)  exact finite-shot mean squared error over the observable basket, the
          pre-registered Family B formula (PREREG_holdout3.md, Section 4):
            MSE_k(N) = mean_S [ (mu_kS - mu*_S)^2 + max(gamma_k^2 - mu_kS^2, 0) / N ]
          at N = 1e3, 1e4, 1e5, 1e6 shots (decision A6)

Statistics (exploratory): paired Wilcoxon signed-rank over circuits, seed-averaged;
bootstrap 95% CI of the mean paired difference (10 000 resamples, seed 2026);
win / tie / loss counts; Holm over the five metrics within each comparison.
Reported for all 143 circuits, the connected circuits and the separable circuits
(Finding E), and by family and width.

Reproducibility check: the per-circuit gap-weighted tau recomputed here for the
model and GBM must equal runs/rq1_h3/per_circuit.csv; the run stops otherwise.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "datagen", HERE):
    sys.path.insert(0, str(p))

NS = (1e3, 1e4, 1e5, 1e6)
N_BOOT, SEED = 10_000, 2026
TIE_F = 1e-12
METHODS = ["model", "gbm-layout", "esp_layout", "min_overhead", "metis", "random", "find_cuts"]
VERSUS = ["model", "gbm-layout", "esp_layout", "min_overhead"]       # each compared with find_cuts
METRICS = ["F"] + [f"MSE@{int(n):.0e}".replace("+0", "") for n in NS]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mse_vec(mu: np.ndarray, mu_star: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """(K, len(NS)) exact MSE, identical to analysis/budget_h3.py (terms + mse)."""
    b2 = np.mean((mu - mu_star[None, :]) ** 2, axis=1)
    v = np.mean(np.clip(g2[:, None] - mu ** 2, 0.0, None), axis=1)
    return b2[:, None] + v[:, None] / np.asarray(NS)[None, :]


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def paired(d: np.ndarray, rng) -> dict:
    d = np.asarray(d, float)
    out = {"n": int(len(d)), "mean": float(d.mean()) if len(d) else float("nan"),
           "wins": int((d > TIE_F).sum()), "ties": int((np.abs(d) <= TIE_F).sum()),
           "losses": int((d < -TIE_F).sum())}
    if len(d) >= 5 and np.any(np.abs(d) > TIE_F):
        out["p"] = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
        m = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(1)
        out["ci"] = [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]
    else:
        out["p"], out["ci"] = float("nan"), [float("nan")] * 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout3", required=True)
    ap.add_argument("--labels", required=True, help="output directory of findcuts_label.py")
    ap.add_argument("--dryrun", required=True, help="findcuts_dryrun/dryrun_rows.json (separability)")
    ap.add_argument("--rq1h3", required=True, help="runs/rq1_h3 (weights, partB_record, per_circuit.csv)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.environ.get("QUEST_UNBLIND") != "holdout3":
        raise SystemExit("holdout3 is guarded: set QUEST_UNBLIND=holdout3 (exploratory, decision A1)")
    import torch
    from quest_phase2 import baselines as B
    from quest_phase2.data import load_split, stratified_split
    from quest_phase2.metrics import circuit_metrics
    from quest_phase2.models import VARIANTS, GraphTransformer
    from run_rq1 import trainable
    from run_rq1_v2 import predict, standardise_hand

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rq = Path(a.rq1h3)
    record = json.loads((rq / "partB_record.json").read_text())
    for fname, h in record["weights"].items():
        if fname.startswith("gt_size-layout_seed"):
            assert sha256(rq / fname) == h, f"weight file {fname} differs from partB_record.json"
    print("frozen weights verified against partB_record.json", flush=True)

    # ---- graphs exactly as run_rq1_h3.evaluate builds them (layout aggregation)
    pool = load_split(Path(a.train), "train", aggregation="layout")
    tr, va = stratified_split(pool)
    h3 = load_split(Path(a.holdout3), "holdout3", aggregation="layout")
    chosen = set(json.loads((Path(a.holdout3).parent / "holdout3_selection.json").read_text())["circuits"])
    h3 = [g for g in h3 if g.cid in chosen]
    assert len(h3) == len(chosen), f"{len(h3)} loaded vs {len(chosen)} selected"
    standardise_hand(tr, va + h3)
    seeds, hp = record["seeds"], record["hp"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scores = defaultdict(dict)                     # scores[method][seed] = {cid: array}
    for s in seeds:
        m = GraphTransformer(hp["d"], hp["heads"], hp["layers"], dropout=hp["dropout"],
                             **VARIANTS["gt_size"]).to(device)
        m.load_state_dict(torch.load(rq / f"gt_size-layout_seed{s}.pt", map_location=device))
        scores["model"][s] = {g.cid: predict(m, g, device) for g in h3}
        gbm = B.GBMRanker(s).fit([g for g in tr if trainable(g)])
        scores["gbm-layout"][s] = {g.cid: gbm.score(g) for g in h3}
    for name, fn in (("esp_layout", B.score_esp), ("min_overhead", B.score_min_overhead),
                     ("metis", B.score_metis)):
        scores[name][0] = {g.cid: fn(g) for g in h3}

    # ---- reproducibility check against the 16 Oct evaluation
    saved = defaultdict(dict)
    with open(rq / "per_circuit.csv") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "holdout3" and r["model"] in ("gt_size-layout", "gbm-layout") and r["kendall_w"]:
                saved[(r["model"], int(r["seed"]))][r["cid"]] = float(r["kendall_w"])
    diffs = []
    for name, key in (("model", "gt_size-layout"), ("gbm-layout", "gbm-layout")):
        for s in seeds:
            for g in h3:
                kw = circuit_metrics(g.y, scores[name][s][g.cid]).get("kendall_w")
                if kw is not None and g.cid in saved[(key, s)]:
                    diffs.append(abs(kw - saved[(key, s)][g.cid]))
    diffs = np.array(diffs)
    share_diff = float(np.mean(diffs > 1e-6)) if len(diffs) else 1.0
    print(f"reproducibility: {len(diffs)} circuit-seeds compared with the 16 Oct evaluation; "
          f"identical {1 - share_diff:.1%}, max |diff| {diffs.max() if len(diffs) else float('nan'):.2e}", flush=True)
    assert len(diffs) > 0 and share_diff <= 0.01, \
        "recomputed holdout3 scores differ from the saved 16 Oct evaluation in more than 1% of circuit-seeds"

    sep = {r["cid"] for r in json.loads(Path(a.dryrun).read_text()) if r["no_cut_needed"]}
    rows = []
    for g in h3:
        c = json.loads((Path(a.holdout3) / f"{g.cid}.json").read_text())
        lab = json.loads((Path(a.labels) / f"{g.cid}.json").read_text())
        mu_star = np.asarray(c["obs_ideal"], float)
        F = np.array([r["pst_hellinger"] for r in g.records], float)
        mu = np.array([r["obs_mu"] for r in g.records], float)
        g2 = np.array([r["gamma2"] for r in g.records], float)
        M = mse_vec(mu, mu_star, g2)                                   # (K, 4)
        row = {"cid": g.cid, "family": g.family, "n_qubits": g.n_qubits, "separable": g.cid in sep,
               "fc_status": lab["status"], "fc_width": lab["width_used"],
               "fc_reused": lab["reused_candidate_label"], "n_cand": len(F)}
        if lab["status"] != "ok" or lab["label"] is None:
            row["excluded"] = f"find_cuts status {lab['status']}"
            rows.append(row)
            continue
        L = lab["label"]
        F_fc = float(L["pst_hellinger"])
        M_fc = mse_vec(np.asarray(L["obs_mu"], float)[None, :], mu_star, np.array([L["gamma2"]]))[0]
        vals = {"find_cuts": (F_fc, M_fc), "random": (float(F.mean()), M.mean(0))}
        for name in ("model", "gbm-layout", "esp_layout", "min_overhead", "metis"):
            picks = [int(np.argmax(sc[g.cid])) for sc in scores[name].values()]
            vals[name] = (float(np.mean(F[picks])), M[picks].mean(0))
            row[f"{name}_picks"] = picks
        F_or = max(F.max(), F_fc)
        M_or = np.minimum(M.min(0), M_fc)
        for name, (f, mm) in vals.items():
            row[f"{name}|F"] = f
            row[f"{name}|regret"] = F_or - f
            for j, key in enumerate(METRICS[1:]):
                row[f"{name}|{key}"] = float(mm[j])
                row[f"{name}|{key}|ratio_to_oracle"] = float(mm[j] / M_or[j]) if M_or[j] > 0 else float("nan")
        row["oracle|F"] = F_or
        row["fc_gamma2"], row["fc_cut_type"] = float(L["gamma2"]), L.get("cut_type")
        rows.append(row)

    ok = [r for r in rows if "excluded" not in r]
    excluded = [(r["cid"], r["excluded"]) for r in rows if "excluded" in r]
    rng = np.random.default_rng(SEED)
    subsets = {"all": ok, "connected": [r for r in ok if not r["separable"]],
               "separable": [r for r in ok if r["separable"]]}
    for fam in sorted({r["family"] for r in ok}):
        subsets[f"family={fam}"] = [r for r in ok if r["family"] == fam]
    for n in sorted({r["n_qubits"] for r in ok}):
        subsets[f"width={n}"] = [r for r in ok if r["n_qubits"] == n]

    summary, tests = {}, {}
    for sname, rs in subsets.items():
        summary[sname] = {"n": len(rs)}
        for meth in METHODS:
            summary[sname][meth] = {
                "F": float(np.mean([r[f"{meth}|F"] for r in rs])),
                "regret": float(np.mean([r[f"{meth}|regret"] for r in rs])),
                **{f"{k}|geo_ratio_to_oracle": float(np.exp(np.mean(np.log(
                    [r[f"{meth}|{k}|ratio_to_oracle"] for r in rs])))) for k in METRICS[1:]}}
        if sname in ("all", "connected", "separable"):
            tests[sname] = {}
            for meth in VERSUS:
                res = {}
                res["F"] = paired(np.array([r[f"{meth}|F"] - r["find_cuts|F"] for r in rs]), rng)
                for k in METRICS[1:]:   # positive = method has LOWER MSE than find_cuts (better)
                    d = np.array([math.log(r[f"find_cuts|{k}"] / r[f"{meth}|{k}"]) for r in rs])
                    res[k] = paired(d, rng)
                    res[k]["geo_ratio_fc_over_method"] = float(np.exp(d.mean()))
                adj = holm([res[k]["p"] if np.isfinite(res[k]["p"]) else 1.0 for k in METRICS])
                for k, p in zip(METRICS, adj):
                    res[k]["p_holm"] = float(p)
                tests[sname][f"{meth}_vs_find_cuts"] = res

    keys = sorted({k for r in rows for k in r})
    with open(out / "per_circuit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "tests.json").write_text(json.dumps({"excluded": excluded, "tests": tests}, indent=1))

    lines = [f"Card A: comparison with qiskit-addon-cutting find_cuts on holdout3 (EXPLORATORY, decision A1)",
             f"circuits compared: {len(ok)} of {len(rows)}; excluded: {excluded or 'none'}",
             f"find_cuts cut reused from candidates: {sum(r['fc_reused'] for r in ok)}; "
             f"no-cut option: {sum(r['fc_cut_type'] == 'none' for r in ok)}; "
             f"rerun at width < 13: {sum(1 for r in ok if r['fc_width'] < 13)}", ""]
    for sname in ("all", "connected", "separable"):
        s = summary[sname]
        lines.append(f"== {sname} (n = {s['n']})")
        lines.append(f"{'method':<14}{'F':>8}{'regret':>9}" + "".join(f"{'MSE/oracle@' + k[4:]:>17}" for k in METRICS[1:]))
        for meth in METHODS:
            v = s[meth]
            lines.append(f"{meth:<14}{v['F']:>8.4f}{v['regret']:>9.4f}"
                         + "".join(f"{v[k + '|geo_ratio_to_oracle']:>17.2f}" for k in METRICS[1:]))
        lines.append("  paired vs find_cuts (F: +diff = method better; MSE: +log ratio = method lower MSE)")
        for meth in VERSUS:
            t = tests[sname][f"{meth}_vs_find_cuts"]
            parts = []
            for k in METRICS:
                v = t[k]
                eff = f"{v['mean']:+.4f}" if k == "F" else f"x{v['geo_ratio_fc_over_method']:.2f}"
                parts.append(f"{k} {eff} W/T/L {v['wins']}/{v['ties']}/{v['losses']} Holm p {v['p_holm']:.1e}")
            lines.append(f"  {meth}: " + " | ".join(parts))
        lines.append("")
    lines.append("by family and width: summary.json")
    (out / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print("COMPARISON FINISHED")


if __name__ == "__main__":
    main()
