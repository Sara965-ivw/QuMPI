"""Relabel the training circuits under scheme v3.1 (adds the observable basket).

The training split was labelled under v3 (fidelity only, no obs_mu / obs_ideal), so a model
cannot be trained on the error at N shots. This regenerates the same 210 training circuits
(corpus.build_spec("train"), same seeds) and labels them with the frozen thesis labeller
(corpus._label_one -> label_circuit_v3, train settings) into a NEW directory train_v31/.
The original train/ directory is never touched. Resumable: finished circuits are skipped.

  python -m paper_budget.datagen.relabel_train_v31 --estimate     # runtime estimate only
  python -m paper_budget.datagen.relabel_train_v31 --workers 4    # run
  python -m paper_budget.datagen.relabel_train_v31 --check        # compare with the v3 labels
Requires PYTHONHASHSEED=0 (layout ties, see notes/DECISIONS_qdislib.md).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from paper_budget.config import ROOT, DATA_V2

sys.path.insert(0, str(ROOT / "stage2_predictor" / "datagen"))
import corpus  # noqa: E402  (frozen thesis code)

SPLIT = "train_v31"


def estimate():
    walls = [json.loads(p.read_text()).get("circuit_wall_s", 0.0)
             for p in (DATA_V2 / "train").glob("*.json")]
    tot = sum(walls)
    print(f"v3 train labelling: {len(walls)} circuits, {tot / 3600:.2f} CPU-hours, "
          f"longest {max(walls) / 60:.1f} min")
    for w in (2, 4, 6):
        print(f"  ~{tot / w / 3600:.2f} h with {w} workers (lower bound: longest circuit)")


def run(workers):
    assert os.environ.get("PYTHONHASHSEED") == "0", "export PYTHONHASHSEED=0 first"
    (DATA_V2 / SPLIT).mkdir(parents=True, exist_ok=True)
    spec = corpus.build_spec("train")
    jobs = sorted([(cid, fam, kw, SPLIT, str(DATA_V2), {}) for cid, fam, kw in spec],
                  key=lambda j: -j[2]["n_qubits"])
    (DATA_V2 / f"manifest_{SPLIT}.json").write_text(json.dumps(
        {**corpus._manifest("train", str(DATA_V2), corpus.SETTINGS["train"]),
         "label_scheme": "v3.1", "relabel_of": "train", "pythonhashseed": "0"}, indent=1))
    t0 = time.time()
    with Pool(workers, maxtasksperchild=4) as pool:
        for i, (cid, st, wall, n) in enumerate(pool.imap_unordered(corpus._label_one, jobs), 1):
            print(f"[{i:3d}/{len(jobs)}] {st:7s} {cid:<38s} {n:3d} cands {wall:7.1f}s  "
                  f"elapsed {(time.time() - t0) / 60:6.1f} min", flush=True)
    print(f"{SPLIT}: finished in {(time.time() - t0) / 3600:.2f} h")


def check():
    """Same circuits and candidates as v3? Compare candidate identities and fidelity labels."""
    ident = lambda r: json.dumps([r["cut_type"], r["home_a"], r["wire_cuts"], r["gate_cut_indices"]])
    n_same = n_cand_diff = n_fid_diff = n_obs = 0
    dF = []
    for p in sorted((DATA_V2 / SPLIT).glob("*.json")):
        b = json.loads(p.read_text())
        n_obs += "obs_ideal" in b
        ra = json.loads((DATA_V2 / "train" / p.name).read_text()).get("records") or []
        rb = b.get("records") or []
        if [ident(r) for r in ra] != [ident(r) for r in rb]:
            n_cand_diff += 1
            continue
        n_same += 1
        d = [abs(x["pst_hellinger"] - y["pst_hellinger"]) for x, y in zip(ra, rb)]
        dF += d
        n_fid_diff += any(x > 1e-9 for x in d)
    print(f"files: {n_same + n_cand_diff}, with obs_ideal: {n_obs}")
    print(f"same candidates as v3: {n_same}; different candidates: {n_cand_diff}")
    if dF:
        dF = np.array(dF)
        print(f"fidelity |v3.1 - v3| over {len(dF)} candidates: max {dF.max():.2e}, "
              f"share > 1e-9 {np.mean(dF > 1e-9):.3f}; circuits with any difference: {n_fid_diff}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    estimate() if a.estimate else check() if a.check else run(a.workers)
