"""Generate holdout4: the confirmatory test set (notes/PREREG_holdout4.md).

Same procedure as holdout3 (stage2_predictor/datagen/corpus.py: build_spec + run_holdout3, frozen), with
one change: a new, never-used seed range (base 99,000; holdout3 used 97,001-97,540).
- widths 16, 18, 20 x families clustered_random, twolocal, qaoa_sbm = 9 cells;
- per cell, circuits are labelled in seed order until 16 are eligible (>= 2 labelled candidates with
  >= 2 distinct sampling overheads; structure only, never a label value) or 60 seeds are used;
- labelling settings identical to holdout3 (g_max 2, w_max 1, max_width 13, 12 candidates), frozen
  labeller corpus._label_one / label_circuit_v3 (adds the observable basket), PYTHONHASHSEED=0.
build_spec_h4 re-implements corpus.build_spec for this split; test_gen_holdout4.py proves that with the
holdout3 seed base it reproduces corpus.build_spec("holdout3") exactly.
Resumable (finished circuits are skipped). Prints structure only (number of candidates), no labels.

  python -m paper_budget.datagen.gen_holdout4 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from paper_budget.config import ROOT, DATA_V2

sys.path.insert(0, str(ROOT / "stage2_predictor" / "datagen"))
import corpus  # noqa: E402  (frozen thesis code)

SPLIT = "holdout4"
SEED_BASE_H4 = 99_000
SETTINGS_H4 = dict(corpus.SETTINGS["holdout3"])


def build_spec_h4(seed_base: int = SEED_BASE_H4, split: str = SPLIT) -> list:
    """corpus.build_spec logic for a holdout3-style split (60 seeds per cell)."""
    spec, seed = [], seed_base
    for n in corpus.HOLDOUT_WIDTHS:
        for fam in corpus.FAMILIES:
            for _ in range(corpus.H3_MAX_ATTEMPTS):
                seed += 1
                r = np.random.default_rng(seed)
                nb = int(r.choice([2, 3])) if n >= 12 else 2
                if fam == "clustered_random":
                    kw = dict(n_qubits=n, depth=int(r.choice([3, 4, 5])), n_blocks=nb,
                              p_bridge=float(r.choice([0.2, 0.4, 0.6])), seed=seed)
                elif fam == "twolocal":
                    kw = dict(n_qubits=n, reps=int(r.choice([1, 2, 3])), seed=seed,
                              entanglement="linear")
                else:
                    kw = dict(n_qubits=n, p_layers=int(r.choice([1, 2])), n_blocks=nb,
                              p_in=float(r.choice([0.4, 0.6])),
                              n_inter=int(r.choice([1, 2, 3])), seed=seed)
                spec.append((f"{split}_{fam}_n{n:02d}_s{seed}", fam, kw))
    return spec


def run(workers: int) -> None:
    assert os.environ.get("PYTHONHASHSEED") == "0", "export PYTHONHASHSEED=0 first"
    root = DATA_V2 / SPLIT
    root.mkdir(parents=True, exist_ok=True)
    spec = build_spec_h4()
    man = DATA_V2 / f"manifest_{SPLIT}.json"
    if not man.exists():
        m = corpus._manifest("holdout3", str(DATA_V2), SETTINGS_H4)
        m.update(split=SPLIT, seed_base=SEED_BASE_H4, label_scheme="v3.1", n_circuits=len(spec),
                 h4_target_per_cell=corpus.H3_TARGET, h4_max_attempts=corpus.H3_MAX_ATTEMPTS,
                 procedure="corpus.run_holdout3 (sequential sampling), seed base 99000",
                 pythonhashseed=os.environ.get("PYTHONHASHSEED"))
        man.write_text(json.dumps(m, indent=1))
    cells = {}
    for cid, fam, kw in spec:
        cells.setdefault((kw["n_qubits"], fam), []).append((cid, fam, kw))
    t0, rnd = time.time(), 0
    while True:
        jobs = []
        for cell, items in sorted(cells.items()):
            elig = [c for c in items if (root / f"{c[0]}.json").exists()
                    and corpus.h3_eligible(root / f"{c[0]}.json")]
            need = corpus.H3_TARGET - len(elig)
            if need <= 0:
                continue
            todo = [c for c in items if not (root / f"{c[0]}.json").exists()]
            jobs += [(cid, fam, kw, SPLIT, str(DATA_V2), dict(SETTINGS_H4)) for cid, fam, kw in todo[:need]]
        if not jobs:
            break
        rnd += 1
        print(f"--- holdout4 round {rnd}: {len(jobs)} circuits", flush=True)
        jobs.sort(key=lambda j: -j[2]["n_qubits"])
        with Pool(workers, maxtasksperchild=2) as pool:
            for i, (cid, st, wall, ncand) in enumerate(pool.imap_unordered(corpus._label_one, jobs), 1):
                print(f"[{i:3d}/{len(jobs)}] {st:7s} {cid:<40s} {ncand:3d} cands {wall:7.1f}s  "
                      f"elapsed {(time.time() - t0) / 60:6.1f} min", flush=True)
    selection, report = [], {}
    for cell, items in sorted(cells.items()):
        chosen, attempts = [], 0
        for cid, _, _ in items:
            f = root / f"{cid}.json"
            if not f.exists():
                break
            attempts += 1
            if corpus.h3_eligible(f):
                chosen.append(cid)
                if len(chosen) == corpus.H3_TARGET:
                    break
        selection += chosen
        report[f"n{cell[0]}_{cell[1]}"] = {"eligible": len(chosen), "attempts": attempts}
        print(f"cell n={cell[0]:2d} {cell[1]:<17s} eligible {len(chosen):2d}/{corpus.H3_TARGET} after {attempts} seeds")
    (DATA_V2 / f"{SPLIT}_selection.json").write_text(json.dumps(
        {"rule": f"first {corpus.H3_TARGET} eligible per cell in seed order, max {corpus.H3_MAX_ATTEMPTS} seeds",
         "n_selected": len(selection), "cells": report, "circuits": selection}, indent=1))
    print(f"holdout4: {len(selection)} circuits selected, {(time.time() - t0) / 3600:.2f} h "
          f"(labels NOT summarised: blinded until the pre-registered evaluation)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    run(ap.parse_args().workers)
