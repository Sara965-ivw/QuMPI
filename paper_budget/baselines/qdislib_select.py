"""Select the Qdislib cut per circuit and seed, and check S1-S3 (thesis environment).

Decisions: notes/DECISIONS_qdislib.md (Q2 width rule, Q3 unrepresentable cuts, amendment 1).
For every circuit and seed, the default find_cut results are tried from width 13 down;
the first that fits after routing is kept (decision A5 for find_cuts). SparseCut is tried
from max_cuts 1 up. Feasibility uses the same functions as find_cuts' find_feasible:
  no cut needed -> nocut_fits; gate cuts only -> multipiece_fits; with a wire cut -> routed_fits.
A cut the labellers cannot represent, or with more than MAX_GATE_CUTS gate cuts
(amendment 1), stops the search for that seed and is kept for the optimistic bound.
No label is read here: S1-S3 are structural.
"""
import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from multiprocessing import Pool

from paper_budget.config import ROOT, DATA_V2, RUNS, DEVICE
from paper_budget.qdislib_map import map_qdislib, pieces_after_gate_cuts

sys.path.insert(0, str(ROOT / "stage2_predictor" / "analysis"))
import findcuts_map as F  # noqa: E402
from quest_phase2.data import rebuild_circuit  # noqa: E402
from quest_phase1.mixed_cuts import LOG2_GAMMA2_WIRE, gate_cut_gamma, plan_cut  # noqa: E402
from quest_phase1.multipiece import multipiece_fits  # noqa: E402

SEEDS = (1234, 2027, 3141, 4242, 5555)
MAX_GATE_CUTS = 6            # amendment 1: 6^g QPD templates; beyond this, bound only
CUTS = RUNS / "qdislib" / "cuts"
OUT = RUNS / "qdislib" / "select"


class CircuitTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise CircuitTimeout()


def log2_gamma2(circ, m):
    """Sampling overhead under the labeller's model: gate cuts (1 + 2|sin theta|)^2, wire 16."""
    g = sum(2 * math.log2(gate_cut_gamma(circ.gates[i].name, circ.gates[i].params))
            for i in m.gate_cut_indices)
    return g + LOG2_GAMMA2_WIRE * m.n_wire_cuts


def check(ctx, circ, result, cache):
    """One find_cut result -> status in {ok, too_wide, unrepresentable, too_many_cuts, error}.
    The logical width is checked first: a cut that leaves a piece wider than the ceiling is
    too_wide (the search goes on), never an error."""
    try:
        m = map_qdislib(circ, result)
    except Exception as exc:
        return {"status": "error", "reason": f"convert: {type(exc).__name__}: {exc}"}
    info = {"gate_cut_indices": m.gate_cut_indices, "wire_cuts": [list(w) for w in m.wire_cuts],
            "n_fragments": m.n_fragments, "home_a": m.home_a, "no_cut": m.no_cut_needed,
            "log2_gamma2": 0.0 if m.no_cut_needed else log2_gamma2(circ, m), "reason": m.reason}
    key = (tuple(sorted(m.gate_cut_indices)), tuple(sorted(map(tuple, m.wire_cuts))))
    if key not in cache:
        cache[key] = _feasibility(ctx, circ, m)
    return dict(info, **cache[key])


def _feasibility(ctx, circ, m):
    if m.no_cut_needed:
        ok, det = F.nocut_fits(ctx, circ)
        return {"status": "ok" if ok else "too_wide", "kind": "nocut", "routed": det}
    if m.n_wire_cuts == 0:
        widths, internal = pieces_after_gate_cuts(circ, m.gate_cut_indices)
        base = {"kind": "multipiece", "piece_widths": widths}
        if widths[0] > F.QPU_WIDTH:
            return dict(base, status="too_wide", reason=f"logical piece of {widths[0]} qubits")
        if internal:
            return dict(base, status="unrepresentable", reason=f"non-separating gate cuts {internal}")
        if len(m.gate_cut_indices) > MAX_GATE_CUTS:
            return dict(base, status="too_many_cuts")
        try:
            ok, det = multipiece_fits(ctx, circ, m.gate_cut_indices)
        except Exception as exc:
            return dict(base, status="error", reason=f"{type(exc).__name__}: {exc}")
        return dict(base, status="ok" if ok else "too_wide", routed=det)
    base = {"kind": "two_side"}
    if m.reason.startswith("fragment width"):
        return dict(base, status="too_wide", reason=m.reason)
    if not m.representable or m.n_fragments != 2:
        return dict(base, status="unrepresentable", reason=m.reason)
    if len(m.gate_cut_indices) > MAX_GATE_CUTS:
        return dict(base, status="too_many_cuts")
    ok, det = F.routed_fits(ctx, circ, plan_cut(circ, F.as_mixed_cut(circ, m)))
    return dict(base, status="ok" if ok else "too_wide", routed=det)


def select(ctx, circ, calls, mode, order_key, cache):
    """First feasible result in the given order; stop at a result that cannot be labelled."""
    tried = []
    for c in sorted(calls, key=order_key):
        if c["error"]:                                   # a failed Qdislib call is never read as "no cut"
            r = {"status": "error", "reason": f"qdislib: {c['error']}"}
        else:
            r = check(ctx, circ, c["result"], cache)
        r.update(width=c.get("max_qubits"), max_cuts=c.get("max_cuts"))
        tried.append(r["status"])
        if r["status"] in ("ok", "unrepresentable", "too_many_cuts", "error"):
            return dict(r, tried=tried)
    return {"status": "infeasible", "tried": tried}


def one_circuit(cid, time_limit):
    from quest_phase1.labeling_v2 import make_device
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(time_limit)
    t0 = time.time()
    row = {"cid": cid, "default": {}, "sparsecut": None, "status": "done"}
    try:
        ctx = make_device(DEVICE, seed=1234)
        d = json.loads((DATA_V2 / "holdout3" / f"{cid}.json").read_text())
        circ = rebuild_circuit(d["circuit_meta"])
        calls = json.loads((CUTS / f"{cid}.json").read_text())["calls"]
        cache = {}
        for s in SEEDS:
            mine = [c for c in calls if c["mode"] == "default" and c["seed"] == s]
            row["default"][str(s)] = select(ctx, circ, mine, "default", lambda c: -c["max_qubits"], cache)
        sparse = [c for c in calls if c["mode"] == "sparsecut"]
        row["sparsecut"] = select(ctx, circ, sparse, "sparsecut", lambda c: c["max_cuts"], cache)
    except CircuitTimeout:
        row["status"] = f"timeout after {time_limit} s"
    except Exception as exc:
        row["status"] = f"error: {type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
    row["wall_s"] = round(time.time() - t0, 1)
    (OUT / f"{cid}.json").write_text(json.dumps(row, indent=1, default=str))
    return cid, row["status"], row["wall_s"]


def _run(args):
    return one_circuit(*args)


def summarise():
    fc = {r["cid"]: r for r in csv.DictReader(open(ROOT / "runs" / "findcuts_compare" / "per_circuit.csv"))}
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("holdout3_*.json"))]
    print(f"\ncircuits selected: {len(rows)}  not done: {[r['cid'] for r in rows if r['status'] != 'done']}")
    pairs = [(r["cid"], s, v) for r in rows if r["status"] == "done" for s, v in r["default"].items()]
    status = {}
    for _, _, v in pairs:
        status[v["status"]] = status.get(v["status"], 0) + 1
    print(f"default find_cut, (circuit, seed) pairs by status: {status}")

    n_err = status.get("error", 0)
    if n_err:
        print(f"WARNING: {n_err} default pairs ended in an error; S1-S2 exclude them "
              f"and cannot be judged until the errors are explained")

    sep = [(c, v) for c, _, v in pairs if fc[c]["separable"] == "True" and v["status"] != "error"]
    cut = [v for _, v in sep if v["status"] != "infeasible" and not v.get("no_cut")]
    print(f"\nS1 separable: Qdislib returns a cut in {len(cut)} of {len(sep)} pairs "
          f"({len(cut) / max(len(sep), 1):.3f}); prediction: > 0.5")

    ge = eq = lt = 0
    ratios = []
    for c, _, v in pairs:
        if fc[c]["separable"] == "True" or v["status"] != "ok" or fc[c]["fc_status"] != "ok":
            continue
        a, b = v["log2_gamma2"], math.log2(float(fc[c]["fc_gamma2"]))
        ratios.append(2 ** (a - b))
        if abs(a - b) < 1e-9:
            eq += 1
        elif a > b:
            ge += 1
        else:
            lt += 1
    n = ge + eq + lt
    ratios.sort()
    print(f"S2 connected, both feasible: {n} pairs; Qdislib gamma^2 greater {ge}, equal {eq}, "
          f"smaller {lt}; share >= {(ge + eq) / max(n, 1):.3f} (prediction: >= 0.90); "
          f"median ratio {ratios[len(ratios) // 2] if ratios else float('nan'):.3g}")

    sp = [r["sparsecut"]["status"] for r in rows if r["status"] == "done"]
    checked = [s for s in sp if s != "error"]
    print(f"S3 SparseCut statuses: { {s: sp.count(s) for s in sorted(set(sp))} }; "
          f"share without a feasible cut among {len(checked)} checked circuits: "
          f"{sum(s != 'ok' for s in checked) / max(len(checked), 1):.3f} (prediction: > 0.5)")
    if len(checked) < len(sp):
        print(f"WARNING: {len(sp) - len(checked)} SparseCut errors are excluded, not counted as failures")


def main():
    assert os.environ.get("QUEST_UNBLIND") == "holdout3", "set QUEST_UNBLIND=holdout3"
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--time-limit", type=int, default=1800, help="seconds per circuit")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        cids = sorted(p.stem for p in CUTS.glob("holdout3_*.json"))[:args.limit]
        todo = [c for c in cids if not (OUT / f"{c}.json").exists()]
        print(f"{len(todo)} of {len(cids)} circuits to select ({len(cids) - len(todo)} already done)")
        with Pool(args.workers) as pool:
            for cid, st, sec in pool.imap_unordered(_run, [(c, args.time_limit) for c in todo]):
                print(f"{cid}: {st} ({sec} s)", flush=True)
    summarise()


if __name__ == "__main__":
    main()
