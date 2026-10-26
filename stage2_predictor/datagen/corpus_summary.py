"""Datasheet statistics for a labelled corpus split (scheme v3).

Usage: python corpus_summary.py --corpus DIR/train --out OUTDIR
Writes OUTDIR/<split>_records.csv (one row per labelled cut),
OUTDIR/<split>_circuits.csv (one row per circuit) and prints the summary.
"""
import argparse
import os
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

REC_COLS = ["circuit_id", "family", "n_qubits", "cut_type", "g", "w", "k", "width_a", "width_b",
            "pst_hellinger", "tv_distance", "f_uncut", "delta_vs_uncut", "routing_overhead_2q",
            "log2_sampling_overhead", "norm_residue", "negative_mass", "oracle_tvd",
            "runs_a", "runs_b", "label_wall_s"]
CIRC_COLS = ["circuit_id", "family", "n_qubits", "n_candidates_total", "n_candidates_labelled",
             "n_excluded_mapped_width", "n_oracle_checked", "f_uncut", "circuit_wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    split = Path(a.corpus).name
    if split == "holdout3" and os.environ.get("QUEST_UNBLIND") != "holdout3":
        raise SystemExit("holdout3 is blinded (PREREG_holdout3.md): labels are not summarised "
                         "before unblinding. Set QUEST_UNBLIND=holdout3 only after the "
                         "pre-registration Part B commit.")
    files = sorted(Path(a.corpus).glob("*.json"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs, circs = [], []
    for f in files:
        c = json.loads(f.read_text())
        m = c["circuit_meta"]
        cid, fam, n = m.get("circuit_id", f.stem), m["family"], m["n_qubits"]
        circs.append({"circuit_id": cid, "family": fam, "n_qubits": n,
                      "n_candidates_total": c.get("n_candidates_total"),
                      "n_candidates_labelled": c.get("n_candidates_labelled"),
                      "n_excluded_mapped_width": c.get("n_excluded_mapped_width"),
                      "n_oracle_checked": c.get("n_oracle_checked"),
                      "f_uncut": c.get("f_uncut"), "circuit_wall_s": c.get("circuit_wall_s")})
        for r in c["records"]:
            row = {k: r.get(k) for k in REC_COLS}
            row.update(circuit_id=cid, family=fam, n_qubits=n)
            recs.append(row)
    for name, rows, cols in ((f"{split}_records.csv", recs, REC_COLS),
                             (f"{split}_circuits.csv", circs, CIRC_COLS)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    print(f"=== {split}: {len(circs)} circuits, {len(recs)} labelled cuts ===")
    zero = [c["circuit_id"] for c in circs if not c["n_candidates_labelled"]]
    print(f"circuits with 0 labelled cuts: {len(zero)}")
    for z in zero:
        print("   ", z)
    if not recs:
        print("no labelled cuts yet in this split")
        return
    print("cuts by type:", dict(Counter(r["cut_type"] for r in recs)))
    print("cuts by (g, w):", dict(sorted(Counter((r["g"], r["w"]) for r in recs).items())))
    by_fam_n = defaultdict(list)
    for r in recs:
        by_fam_n[(r["family"], r["n_qubits"])].append(r["pst_hellinger"])
    print(f"{'family':<18}{'n':>4}{'cuts':>6}{'min':>8}{'mean':>8}{'max':>8}")
    for (fam, n), v in sorted(by_fam_n.items()):
        print(f"{fam:<18}{n:>4}{len(v):>6}{min(v):>8.4f}{st.mean(v):>8.4f}{max(v):>8.4f}")
    # variance decomposition (the v1 failure check)
    by_c = defaultdict(list)
    for r in recs:
        by_c[r["circuit_id"]].append(r["pst_hellinger"])
    allv = [r["pst_hellinger"] for r in recs]
    tot = st.pvariance(allv)
    within = sum(st.pvariance(v) * len(v) for v in by_c.values() if len(v) > 1) / len(allv)
    spreads = [max(v) - min(v) for v in by_c.values() if len(v) > 1]
    print(f"label range [{min(allv):.4f}, {max(allv):.4f}]; within-circuit variance share "
          f"{within / tot:.3f} (v1 0.001, v2 pilot 0.098); median within-circuit spread "
          f"{st.median(spreads):.4f}")
    ot = [r["oracle_tvd"] for r in recs if r["oracle_tvd"] is not None]
    print(f"oracle checks: {len(ot)}, worst TVD {max(ot):.1e}")
    nr = [abs(r["norm_residue"]) for r in recs if r["norm_residue"] is not None]
    nm = [r["negative_mass"] for r in recs if r["negative_mass"] is not None]
    print(f"|norm residue| median {st.median(nr):.1e} max {max(nr):.1e}; "
          f"negative mass median {st.median(nm):.1e} max {max(nm):.1e}")
    better = sum(1 for r in recs if r["delta_vs_uncut"] is not None and r["delta_vs_uncut"] > 0)
    have = sum(1 for r in recs if r["delta_vs_uncut"] is not None)
    print(f"cuts better than the uncut circuit: {better}/{have}")
    cpu = sum(c["circuit_wall_s"] or 0 for c in circs) / 3600
    print(f"total labelling CPU time {cpu:.1f} h; files -> {out}")


if __name__ == "__main__":
    main()
