"""Label the feasible Qdislib cuts (thesis environment).

Same labeller, device (fake_brisbane), noise model and transpiler seed (1234) as every
other label. A labelled holdout3 candidate is reused only for the same execution: the same
gate and wire cuts with exactly two subcircuits (the rule of the find_cuts labelling).
Otherwise gate-cut-only cuts are labelled with quest_phase1.multipiece, each piece simulated
on its own. Each circuit runs with a time limit; a timeout is recorded, never hidden.
Only counts and timing are printed; no fidelity or MSE is read here.

Holdout4 (11 Nov 2026, notes/PREDICTIONS_compare_holdout4.md): --split holdout4 reads
runs/paper_budget/qdislib_holdout4/select and writes .../labels. Requires PYTHONHASHSEED=0
(multipiece layout ties, notes/RESULTS_findcuts_relabel.md). The default (--split holdout3)
is unchanged.
"""
import argparse
import json
import os
import signal
import sys
import time
from collections import Counter
from multiprocessing import Pool

from paper_budget.config import ROOT, DATA_V2, RUNS, DEVICE

sys.path.insert(0, str(ROOT / "stage2_predictor"))
sys.path.insert(0, str(ROOT / "stage2_predictor" / "analysis"))
import quest_phase2.data  # noqa: E402,F401  (puts the labeller package on sys.path)
from findcuts_label import LABEL_KEYS  # noqa: E402  (same stored fields as find_cuts labels)



def dirs(split: str):
    """(select, labels) folders: the October layout for holdout3, qdislib_<split>/ otherwise."""
    base = RUNS / ("qdislib" if split == "holdout3" else f"qdislib_{split}")
    return base / "select", base / "labels"


class CircuitTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise CircuitTimeout()


def key(gates, wires):
    return (tuple(sorted(gates)), tuple(sorted(tuple(w) for w in wires)))


def plan(cid, split="holdout3"):
    """Distinct feasible Qdislib cuts of one circuit, and where a label already exists."""
    sel = json.loads((dirs(split)[0] / f"{cid}.json").read_text())
    recs = json.loads((DATA_V2 / split / f"{cid}.json").read_text())["records"]
    cand = {key(r["gate_cut_indices"], r["wire_cuts"]): r for r in recs}
    cuts = {}
    picks = [(f"default_{s}", v) for s, v in sel["default"].items()] + [("sparsecut", sel["sparsecut"])]
    for name, v in picks:
        if v["status"] == "ok":
            k = key(v["gate_cut_indices"], v["wire_cuts"])
            cuts.setdefault(k, {"v": v, "used_by": []})["used_by"].append(name)
    for k, c in cuts.items():
        same = k in cand and c["v"]["n_fragments"] == 2
        c["source"] = "candidate" if same else ("new" if c["v"]["kind"] == "multipiece" else "unsupported")
        c["record"] = cand[k] if same else None
    return cuts


def label_circuit(cid, time_limit, split="holdout3"):
    from quest_phase1.labeling_v2 import make_device
    from quest_phase1.multipiece import label_multipiece
    from quest_phase1.simulators import probabilities
    from quest_phase2.data import rebuild_circuit
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(time_limit)
    t0 = time.time()
    row = {"cid": cid, "status": "done", "cuts": []}
    out_dir = dirs(split)[1]
    try:
        c = json.loads((DATA_V2 / split / f"{cid}.json").read_text())
        circ = rebuild_circuit(c["circuit_meta"])
        ctx = make_device(DEVICE, seed=1234)
        p_ideal = probabilities(circ)
        spec = [tuple(x) for x in c["obs_spec"]]
        for cut in plan(cid, split).values():
            v, t1 = cut["v"], time.time()
            entry = {"used_by": cut["used_by"], "source": cut["source"], "kind": v["kind"],
                     "gate_cut_indices": v["gate_cut_indices"], "wire_cuts": v["wire_cuts"],
                     "n_fragments": v["n_fragments"], "label": None}
            if cut["source"] == "candidate":
                rec = cut["record"]
            elif cut["source"] == "new":
                rec = label_multipiece(ctx, circ, v["gate_cut_indices"], p_ideal=p_ideal,
                                       obs_spec=spec, obs_ideal=c["obs_ideal"])
            else:
                rec = None
            if rec is not None:
                entry["label"] = {k: rec[k] for k in LABEL_KEYS if k in rec}
            entry["seconds"] = round(time.time() - t1, 1)
            row["cuts"].append(entry)
    except CircuitTimeout:
        row["status"] = f"timeout after {time_limit} s"
    except Exception as exc:
        row["status"] = f"error: {type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
    row["wall_s"] = round(time.time() - t0, 1)
    tmp = out_dir / f"{cid}.tmp"
    tmp.write_text(json.dumps(row, indent=1, default=str))
    tmp.replace(out_dir / f"{cid}.json")
    src = Counter(e["source"] for e in row["cuts"])
    return cid, row["status"], dict(src), row["wall_s"]


def _run(args):
    return label_circuit(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="holdout3", choices=("holdout3", "holdout4"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--time-limit", type=int, default=1800, help="seconds per circuit")
    ap.add_argument("--dry-run", action="store_true", help="list what would be labelled, label nothing")
    args = ap.parse_args()
    assert os.environ.get("QUEST_UNBLIND") == args.split, f"set QUEST_UNBLIND={args.split}"
    if args.split != "holdout3":
        assert os.environ.get("PYTHONHASHSEED") == "0", "export PYTHONHASHSEED=0 first"
    sel_dir, out_dir = dirs(args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    cids = sorted(p.stem for p in sel_dir.glob(f"{args.split}_*.json"))[:args.limit]

    if args.dry_run:
        src, gates = Counter(), Counter()
        for cid in cids:
            for c in plan(cid, args.split).values():
                src[c["source"]] += 1
                if c["source"] == "new":
                    gates[len(c["v"]["gate_cut_indices"])] += 1
        print(f"circuits {len(cids)}; distinct cuts by source {dict(src)}")
        print(f"new labels by number of gate cuts (cost ~ 6^g templates): {dict(sorted(gates.items()))}")
        return

    todo = [c for c in cids if not (out_dir / f"{c}.json").exists()]
    print(f"{len(todo)} of {len(cids)} circuits to label", flush=True)
    with Pool(args.workers) as pool:
        for k, (cid, st, src, sec) in enumerate(
                pool.imap_unordered(_run, [(c, args.time_limit, args.split) for c in todo]), start=1):
            print(f"[{k}/{len(todo)}] {cid}: {st} {src} ({sec} s)", flush=True)


if __name__ == "__main__":
    main()
