"""The 10-circuit Phase 1 pilot (Decision Record §5 exit criteria).

Usage:
    python3 pilot_run.py                                   # fixture snapshot, numpy engine
    python3 pilot_run.py --snapshot calibration_snapshots/ibm_brisbane_<date>.json
    python3 pilot_run.py --engine aer                      # after test_bridge.py is green
    python3 pilot_run.py --engine aer --device GPU         # after GPU verification

Produces pilot_results.json and prints the exit-criteria checklist:
  [1] oracle exactness (asserted on every candidate — run aborts if violated)
  [2] label histogram + compression verdict (F3)
  [3] additivity probe pairs (F2)
  [4] wall-time extrapolation to the 200-300-circuit corpus
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quest_phase1.generators import (qaoa_maxcut_circuit, random_circuit,
                                     twolocal_circuit)
from quest_phase1.labeling import additivity_probe, run_pilot

FIXTURE = Path(__file__).resolve().parent / "quest_phase1" / "snapshot_fixture.json"

# The pilot batch: 10 circuits across the three families (Roadmap SS4.2),
# sized for the 8-qubit fixture; regenerate at larger n with a real
# 127-qubit brisbane snapshot. Seeds are the dataset's reproducibility root.
PILOT = [
    (random_circuit, dict(n_qubits=6, depth=4, density=0.5, seed=1001)),
    (random_circuit, dict(n_qubits=7, depth=4, density=0.9, seed=1002)),
    (random_circuit, dict(n_qubits=8, depth=3, density=0.7, seed=1003)),
    (random_circuit, dict(n_qubits=8, depth=5, density=0.5, seed=1004)),
    (twolocal_circuit, dict(n_qubits=6, reps=2, seed=2001)),
    (twolocal_circuit, dict(n_qubits=7, reps=3, seed=2002)),
    (twolocal_circuit, dict(n_qubits=8, reps=2, seed=2003)),
    (qaoa_maxcut_circuit, dict(n_qubits=6, p_layers=1, edge_prob=0.5, seed=3001)),
    (qaoa_maxcut_circuit, dict(n_qubits=7, p_layers=2, edge_prob=0.4, seed=3002)),
    (qaoa_maxcut_circuit, dict(n_qubits=8, p_layers=1, edge_prob=0.35, seed=3003)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(FIXTURE))
    ap.add_argument("--out", default="pilot_results.json")
    ap.add_argument("--kmax", type=int, default=2)
    ap.add_argument("--engine", choices=["numpy", "aer"], default="numpy")
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()

    if args.engine == "aer":
        # Only after tests/test_bridge.py is green on this environment.
        from quest_phase1.qiskit_bridge import aer_density_matrix
        import quest_phase1.labeling as labeling_mod
        import quest_phase1.combiner as combiner_mod
        dm = lambda c, noise=None, init_psi=None: aer_density_matrix(
            c, noise, init_psi=init_psi, device=args.device)
        # route the engine through the labeler
        orig = combiner_mod.label_candidate
        combiner_mod_label = lambda *a, **kw: orig(*a, dm_func=dm, **kw)
        labeling_mod.label_candidate = combiner_mod_label

    circuits = [gen(**kw) for gen, kw in PILOT]
    t0 = time.time()
    out = run_pilot(circuits, args.snapshot, args.out, k_max=args.kmax)
    wall = time.time() - t0
    s = out["summary"]

    print("=" * 66)
    print("PHASE 1 PILOT — exit-criteria checklist")
    print("=" * 66)
    print(f"[1] Oracle exactness: PASSED on all {s['n_labelled_candidates']} "
          f"candidates (asserted inline; run would have aborted otherwise)")

    print(f"[2] Labels: n={s['n_labelled_candidates']}  mean={s['label_mean']:.4f}  "
          f"std={s['label_std']:.4f}  range=[{s['label_min']:.4f}, {s['label_max']:.4f}]")
    compressed = s["label_min"] > 0.95 or s["label_std"] < 0.01
    print("    F3 verdict:", "COMPRESSED near 1 -> decide logit transform / "
          "ranking loss BEFORE scaling (Decision Record F3)" if compressed
          else "usable dynamic range at this scale")
    print("    histogram counts:", s["label_histogram"]["counts"])

    pairs = additivity_probe(out["records"])
    if pairs:
        k1 = np.array([p["pst_k1"] for p in pairs])
        k2 = np.array([p["pst_k2"] for p in pairs])
        rho = float(np.corrcoef(k1, k2)[0, 1]) if len(pairs) > 2 else float("nan")
        print(f"[3] Additivity probe (F2): {len(pairs)} nested (k=1 within k=2) pairs; "
              f"corr(PST_k1, PST_k2) = {rho:.3f}")
        print("    (full additivity analysis belongs in the datasheet; "
              "low correlation would challenge per-cut label composability)")
    else:
        print("[3] Additivity probe (F2): no nested pairs found — widen kmax or batch")

    per_cand = s["total_label_wall_s"] / s["n_labelled_candidates"]
    est = 300 * (s["n_labelled_candidates"] / len(circuits)) * per_cand / 3600
    print(f"[4] Cost: {per_cand:.2f} s/candidate here; naive extrapolation to 300 "
          f"circuits ~ {est:.1f} h on THIS engine/size. CAUTION: real corpus "
          f"circuits are larger (up to 20q); treat as a lower bound and re-measure "
          f"at target sizes before committing the generation run.")
    print(f"Total pilot wall time: {wall:.1f} s  |  results -> {args.out}")
    print("=" * 66)


if __name__ == "__main__":
    main()
