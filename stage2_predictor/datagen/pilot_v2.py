"""Phase 1 pilot under labelling scheme v2 (same 10 circuits as pilot_run.py).

Usage:  python pilot_v2.py --out pilot_v2.json [--workers 10] [--device fake_brisbane]

Prints the decision quantities for the v2 exit criterion:
  * within-circuit share of label variance (v1 pilot: 0.001)
  * per-circuit label spread vs the uncut reference
  * correlation between routing overhead and label (mechanism check)
  * cost per candidate, for the corpus budget
"""
import argparse
import json
import statistics as st
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot_run import PILOT  # identical circuit batch and seeds
from quest_phase1.labeling_v2 import label_circuit_v2, make_device


def _work(job):
    idx, device, seed, kmax = job
    gen, kw = PILOT[idx]
    circ, meta = gen(**kw)
    ctx = make_device(device, seed=seed)
    t0 = time.perf_counter()
    out = label_circuit_v2(ctx, circ, meta, k_max=kmax)
    out["circuit_wall_s"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pilot_v2.json")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default="fake_brisbane")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--kmax", type=int, default=2)
    args = ap.parse_args()

    t0 = time.time()
    jobs = [(i, args.device, args.seed, args.kmax) for i in range(len(PILOT))]
    with Pool(args.workers) as pool:
        circuits = pool.map(_work, jobs)
    wall = time.time() - t0

    recs = [r for c in circuits for r in c["records"]]
    f = [r["pst_hellinger"] for r in recs]
    total_var = st.pvariance(f)
    within = sum(st.pvariance([r["pst_hellinger"] for r in c["records"]]) * len(c["records"])
                 for c in circuits if len(c["records"]) > 1) / len(recs)
    ro = np.array([r["routing_overhead_2q"] for r in recs], float)
    dv = np.array([r["delta_vs_uncut"] for r in recs], float)
    corr = float(np.corrcoef(ro, dv)[0, 1]) if ro.std() > 0 else float("nan")

    summary = {"n_circuits": len(circuits), "n_candidates": len(recs),
               "label_min": min(f), "label_max": max(f), "label_mean": st.mean(f),
               "total_var": total_var, "within_circuit_var": within,
               "within_share": within / total_var if total_var else float("nan"),
               "corr_routing_vs_delta": corr,
               "cpu_s_per_candidate": sum(r["label_wall_s"] for r in recs) / len(recs),
               "wall_s": wall, "device": args.device, "transpiler_seed": args.seed}
    Path(args.out).write_text(json.dumps({"summary": summary, "circuits": circuits}, indent=1))

    print("=" * 70)
    print(f"PILOT v2  device={args.device}  seed={args.seed}  candidates={len(recs)}")
    print("=" * 70)
    print("[1] Mapped oracle: PASSED on every candidate (asserted inline)")
    print(f"[2] Labels range [{min(f):.4f}, {max(f):.4f}]  mean {st.mean(f):.4f}")
    print(f"    within-circuit share of variance: {summary['within_share']:.3f}"
          f"   (v1 pilot: 0.001; v2 must be clearly larger)")
    for c in circuits:
        cf = [r["pst_hellinger"] for r in c["records"]]
        m = c["circuit_meta"]
        print(f"    {m['family']:<12} n={m['n_qubits']}  F_uncut={c['f_uncut']:.4f}  "
              f"cuts=[{min(cf):.4f}, {max(cf):.4f}]  spread={max(cf) - min(cf):.4f}  "
              f"better-than-uncut={sum(x > c['f_uncut'] for x in cf)}/{len(cf)}")
    print(f"[3] corr(routing overhead, label - F_uncut) = {corr:.3f}")
    print(f"[4] {summary['cpu_s_per_candidate']:.2f} CPU-s per candidate; "
          f"wall {wall:.0f} s on {args.workers} workers")
    print(f"results -> {args.out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
