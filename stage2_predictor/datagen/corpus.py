"""Phase 1 corpus generation under labelling scheme v2.

Design (ratified 10 Oct 2026): train on 6-14 qubits, hold out 16-24 qubits,
both with exact noisy-simulation labels. The holdout split uses a disjoint
seed range and is written to a separate directory; it must not be read during
model development (sequestration rule).

Usage
-----
  python corpus.py probe   --out DIR            # ~10 min cost probe, prints a budget
  python corpus.py train   --out DIR [--workers 6]
  python corpus.py holdout --out DIR [--workers 4]

Runs are resumable: each circuit is written to DIR/<split>/<id>.json and
skipped if present. DIR/manifest_<split>.json records the full specification,
device snapshot, seeds and package versions.
"""
import argparse
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quest_phase1 import generators as G
from quest_phase1.labeling_v2 import make_device
from quest_phase1.mixed_cuts import label_circuit_v3

TRAIN_WIDTHS = [6, 8, 10, 12, 14]
HOLDOUT_WIDTHS = [16, 18, 20]       # 22-24 exceed the routed 13-qubit ceiling (probe 10 Oct)
PER_CELL_TRAIN = 14          # per (width, family): 5 x 3 x 14 = 210 circuits
PER_CELL_HOLDOUT = 5         # 3 x 3 x 5 = 45 circuits
SEED_BASE = {"train": 10_000, "holdout": 90_000, "holdout2": 95_000,
             "holdout3": 97_000}                                           # disjoint seed ranges
# holdout3 (pre-registered, PREREG_holdout3.md): sequential sampling per
# (width, family) cell in seed order until H3_TARGET eligible circuits or
# H3_MAX_ATTEMPTS seeds. Eligibility uses structure only (number of labelled
# candidates and distinct sampling overheads), never a fidelity value.
H3_TARGET = 16
H3_MAX_ATTEMPTS = 60

# Scheme v3: mixed wire + gate cuts on cuttable families. Gate-cut terms cost
# 5^g fragment runs, so holdouts (12-13 qubit fragments) are limited to g <= 1.
SETTINGS = {
    "train":   dict(g_max=2, w_max=1, max_width=13, max_candidates=20,
                    oracle_fraction=0.1, uncut_max_qubits=13),
    "holdout": dict(g_max=1, w_max=1, max_width=13, max_candidates=12,
                    oracle_fraction=0.1, uncut_max_qubits=13),
    # fresh confirmatory holdout (after rq1-run1): same widths, new seeds, g <= 2 from the start
    "holdout2": dict(g_max=2, w_max=1, max_width=13, max_candidates=12,
                     oracle_fraction=0.1, uncut_max_qubits=13),
    # second confirmatory holdout: identical labelling settings to holdout2,
    # label scheme v3.1 (adds the fixed observable basket)
    "holdout3": dict(g_max=2, w_max=1, max_width=13, max_candidates=12,
                     oracle_fraction=0.1, uncut_max_qubits=13),
}
FAMILIES = ("clustered_random", "twolocal", "qaoa_sbm")
DEVICE = "fake_brisbane"
TRANSPILER_SEED = 1234


def build_spec(split: str) -> list:
    """Deterministic list of (circuit_id, generator, kwargs)."""
    widths = TRAIN_WIDTHS if split == "train" else HOLDOUT_WIDTHS
    per_cell = {"train": PER_CELL_TRAIN, "holdout3": H3_MAX_ATTEMPTS}.get(split, PER_CELL_HOLDOUT)
    spec, seed = [], SEED_BASE[split]
    for n in widths:
        for fam in FAMILIES:
            for j in range(per_cell):
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


_GEN = {"clustered_random": G.clustered_random_circuit, "twolocal": G.twolocal_circuit,
        "qaoa_sbm": G.qaoa_sbm_circuit}


def _label_one(job):
    cid, fam, kw, split, out_dir, overrides = job
    path = Path(out_dir) / split / f"{cid}.json"
    topup = overrides.pop("_topup", False) if overrides else False
    if path.exists():
        if not topup:
            return cid, "skipped", 0.0, 0
        if json.loads(path.read_text()).get("records"):
            return cid, "kept", 0.0, 0        # already has labels: never relabel
        (path.parent / f"{cid}.g1_empty.json").write_text(path.read_text())  # audit trail
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    circ, meta = _GEN[fam](**kw)
    meta["circuit_id"] = cid
    base = SETTINGS.get(split) or SETTINGS["train" if kw["n_qubits"] <= 14 else "holdout"]
    settings = {**base, **overrides}
    ctx = make_device(DEVICE, seed=TRANSPILER_SEED)
    t0 = time.perf_counter()
    out = label_circuit_v3(ctx, circ, meta, sample_seed=kw["seed"], **settings)
    out["circuit_wall_s"] = time.perf_counter() - t0
    out["split"] = split
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, path)                 # atomic: no half-written files on interrupt
    return cid, "done", out["circuit_wall_s"], len(out["records"])


def _manifest(split, out_dir, settings):
    import qiskit, qiskit_aer, qiskit_ibm_runtime
    ctx = make_device(DEVICE, seed=TRANSPILER_SEED)
    return {"split": split, "device": ctx.name, "calibration_date": ctx.calibration_date,
            "transpiler_seed": TRANSPILER_SEED, "opt_level": ctx.opt_level,
            "settings": settings, "widths": TRAIN_WIDTHS if split == "train" else HOLDOUT_WIDTHS,
            "seed_base": SEED_BASE.get(split),
            "label_scheme": "v3.1" if split == "holdout3" else "v3",
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
                         "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__},
            "n_circuits": len(build_spec(split)) if split in SEED_BASE else None}


def run_split(split, out_dir, workers):
    spec = build_spec(split)
    (Path(out_dir) / split).mkdir(parents=True, exist_ok=True)
    Path(out_dir, f"manifest_{split}.json").write_text(
        json.dumps(_manifest(split, out_dir, SETTINGS[split]), indent=1))
    # largest circuits first so the long jobs start early
    jobs = sorted([(cid, fam, kw, split, out_dir, {}) for cid, fam, kw in spec],
                  key=lambda j: -j[2]["n_qubits"])
    t0, done = time.time(), 0
    with Pool(workers, maxtasksperchild=4) as pool:
        for cid, status, wall, ncand in pool.imap_unordered(_label_one, jobs):
            done += 1
            print(f"[{done:3d}/{len(jobs)}] {status:7s} {cid:<38s} "
                  f"{ncand:3d} cands {wall:7.1f}s  elapsed {(time.time() - t0) / 60:6.1f} min",
                  flush=True)
    print(f"{split}: finished in {(time.time() - t0) / 3600:.2f} h")


def h3_eligible(path: Path) -> bool:
    """Structural eligibility only: >= 2 labelled candidates with >= 2 distinct
    sampling overheads (otherwise there is no choice to rank or to trade off)."""
    d = json.loads(path.read_text())
    recs = d.get("records") or []
    return len(recs) >= 2 and len({round(r["log2_sampling_overhead"], 9) for r in recs}) >= 2


def run_holdout3(out_dir, workers):
    split = "holdout3"
    spec = build_spec(split)
    root = Path(out_dir) / split
    root.mkdir(parents=True, exist_ok=True)
    Path(out_dir, f"manifest_{split}.json").write_text(
        json.dumps({**_manifest(split, out_dir, SETTINGS[split]),
                    "h3_target_per_cell": H3_TARGET, "h3_max_attempts": H3_MAX_ATTEMPTS},
                   indent=1))
    cells = {}
    for cid, fam, kw in spec:                        # spec is in seed order within a cell
        cells.setdefault((kw["n_qubits"], fam), []).append((cid, fam, kw))

    def status(cell):
        done = [c for c in cells[cell] if (root / f"{c[0]}.json").exists()]
        elig = [c for c in done if h3_eligible(root / f"{c[0]}.json")]
        # eligible circuits that precede the first unlabelled seed are final
        return done, elig

    t0, rnd = time.time(), 0
    while True:
        jobs = []
        for cell, items in sorted(cells.items()):
            done, elig = status(cell)
            need = H3_TARGET - len(elig)
            if need <= 0:
                continue
            todo = [c for c in items if not (root / f"{c[0]}.json").exists()]
            # schedule only what is still needed (deviation 13 Oct 2026: the first
            # version queued 2x need, doubling cost in high-yield cells); the
            # selection rule below is unchanged
            jobs += [(cid, fam, kw, split, out_dir, {}) for cid, fam, kw in todo[:need]]
        if not jobs:
            break
        rnd += 1
        print(f"--- holdout3 round {rnd}: {len(jobs)} circuits", flush=True)
        jobs.sort(key=lambda j: -j[2]["n_qubits"])
        with Pool(workers, maxtasksperchild=2) as pool:
            for i, (cid, st, wall, ncand) in enumerate(pool.imap_unordered(_label_one, jobs), 1):
                print(f"[{i:3d}/{len(jobs)}] {st:7s} {cid:<38s} {ncand:3d} cands {wall:7.1f}s  "
                      f"elapsed {(time.time() - t0) / 60:6.1f} min", flush=True)
    # analysis set = first H3_TARGET eligible circuits in seed order per cell
    selection, report = [], {}
    for cell, items in sorted(cells.items()):
        chosen, attempts = [], 0
        for cid, _, _ in items:
            f = root / f"{cid}.json"
            if not f.exists():
                break                                    # seed order: stop at first gap
            attempts += 1
            if h3_eligible(f):
                chosen.append(cid)
                if len(chosen) == H3_TARGET:
                    break
        selection += chosen
        report[f"n{cell[0]}_{cell[1]}"] = {"eligible": len(chosen), "attempts": attempts}
        print(f"cell n={cell[0]:2d} {cell[1]:<17s} eligible {len(chosen):2d}/{H3_TARGET} "
              f"after {attempts} seeds")
    Path(out_dir, "holdout3_selection.json").write_text(json.dumps(
        {"rule": f"first {H3_TARGET} eligible per cell in seed order, max {H3_MAX_ATTEMPTS} seeds",
         "n_selected": len(selection), "cells": report, "circuits": selection}, indent=1))
    print(f"holdout3: {len(selection)} circuits selected, "
          f"{(time.time() - t0) / 3600:.2f} h (labels NOT summarised: blinded until unblinding)")


def topup_holdout(out_dir, workers):
    """Relabel ONLY holdout circuits that have no eligible cut at g <= 1, with
    g <= 2 (the training limit). Circuits that already have labels are kept
    unchanged; the empty g <= 1 result is kept beside the new file for audit."""
    spec = build_spec("holdout")
    jobs = sorted([(cid, fam, kw, "holdout", out_dir, {"g_max": 2, "_topup": True})
                   for cid, fam, kw in spec], key=lambda j: -j[2]["n_qubits"])
    t0, done = time.time(), 0
    with Pool(workers, maxtasksperchild=2) as pool:
        for cid, status, wall, ncand in pool.imap_unordered(_label_one, jobs):
            done += 1
            print(f"[{done:3d}/{len(jobs)}] {status:7s} {cid:<38s} {ncand:3d} cands {wall:7.1f}s  "
                  f"elapsed {(time.time() - t0) / 60:6.1f} min", flush=True)
    print(f"holdout top-up finished in {(time.time() - t0) / 3600:.2f} h")


def probe(out_dir, workers):
    """Label 3 candidates of one circuit per (width, family) and extrapolate."""
    rows = []
    for split in ("train", "holdout"):
        seen = set()
        for cid, fam, kw in build_spec(split):
            key = (kw["n_qubits"], fam)
            if key in seen:
                continue
            seen.add(key)
            rows.append((cid, fam, kw, "probe_v3", out_dir, {"max_candidates": 3,
                                                             "oracle_fraction": 0.34}))
    (Path(out_dir) / "probe_v3").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = sorted(rows, key=lambda j: j[2]["n_qubits"])      # smallest first: early feedback
    width_of = {j[0]: (j[2]["n_qubits"], j[1]) for j in jobs}
    per = {}
    with Pool(workers, maxtasksperchild=2) as pool:
        for i, (cid, status, wall, ncand) in enumerate(pool.imap_unordered(_label_one, jobs), 1):
            if status == "skipped":
                rec = json.loads((Path(out_dir) / "probe_v3" / f"{cid}.json").read_text())
                wall, ncand = rec["circuit_wall_s"], len(rec["records"])
            if ncand:
                per[width_of[cid]] = wall / ncand
            print(f"[{i:2d}/{len(jobs)}] {cid:<38s} {ncand} cands  {wall:7.1f}s  "
                  f"elapsed {(time.time() - t0) / 60:5.1f} min", flush=True)
    print(f"probe wall time {(time.time() - t0) / 60:.1f} min")
    print(f"{'width':>5} {'family':<12} {'s/candidate':>11}")
    for (n, fam), s in sorted(per.items()):
        print(f"{n:5d} {fam:<12} {s:11.1f}")
    for split in ("train", "holdout"):
        cap = SETTINGS[split]["max_candidates"]
        cpu = sum(per.get((kw["n_qubits"], fam), 0.0) * cap
                  for _, fam, kw in build_spec(split))
        w = 6 if split == "train" else 4
        print(f"{split}: ~{cpu / 3600:.1f} CPU-h at cap {cap} -> ~{cpu / 3600 / w:.1f} h wall "
              f"on {w} workers (upper bound: assumes every circuit hits the cap)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "train", "holdout", "holdout2", "holdout3", "topup"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "probe":
        probe(a.out, a.workers or 6)
    elif a.mode == "topup":
        topup_holdout(a.out, a.workers or 4)
    elif a.mode == "holdout3":
        run_holdout3(a.out, a.workers or 6)
    else:
        run_split(a.mode, a.out, a.workers or (8 if a.mode == "train" else 4))


if __name__ == "__main__":
    main()
