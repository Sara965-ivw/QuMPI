"""Run Qdislib find_cut on the exported circuits. Qdislib ENVIRONMENT ONLY.

Decisions: notes/DECISIONS_qdislib.md. Per circuit:
  default    find_cut(max_qubits=w, seed=s), w = 13..5, every seed      (baseline)
  sparsecut  find_cut(coupling_map, layout, max_cuts=k), k = 1..6       (baseline, deterministic)
  top5       find_cut(max_qubits=13, top_n=5, seed=s)                   (candidate union only)
  mixed      find_cut(mixed=True, seed=s)                               (candidate union only)
All widths are computed here; the thesis environment keeps the widest cut that fits
after routing, which is the rule of decision A5 for find_cuts. Every call is timed;
a call over the time limit is recorded as a timeout and the run continues.
"""
import argparse
import json
import signal
import time
from pathlib import Path

import qiskit.qasm2
import Qdislib as qd
from Qdislib.utils.backends import iter_gates

SEEDS = (1234, 2027, 3141, 4242, 5555)


class CallTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise CallTimeout()


def check_numbering(qc, gates):
    """Qdislib's numbering must be exactly the exported gate list, or the run stops."""
    got = [[name.lower(), list(qubits)] for name, qubits in iter_gates(qc)]
    if got != gates:
        where = (next(i for i, (a, b) in enumerate(zip(got, gates)) if a != b)
                 if len(got) == len(gates) else f"length {len(got)} vs {len(gates)}")
        raise AssertionError(f"gate numbering differs from the export at {where}")


def timed_call(qc, limit, **kw):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(limit)
    t0 = time.perf_counter()
    try:
        res, err = qd.find_cut(qc, **kw), None
    except CallTimeout:
        res, err = None, f"timeout after {limit} s"
    except Exception as exc:              # recorded, never hidden
        res, err = None, f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
    return res, err, time.perf_counter() - t0


def plan(pilot):
    widths = [13] if pilot else list(range(13, 4, -1))
    seeds = [1234] if pilot else list(SEEDS)
    calls = [("default", dict(max_qubits=w, seed=s)) for s in seeds for w in widths]
    calls += [("sparsecut", dict(max_cuts=k)) for k in range(1, (1 if pilot else 6) + 1)]
    if not pilot:
        calls += [("top5", dict(max_qubits=13, top_n=5, seed=s)) for s in seeds]
        calls += [("mixed", dict(mixed=True, seed=s)) for s in seeds]
    return calls


def to_json(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default="runs/paper_budget/qdislib/circuits")
    ap.add_argument("--out", default="runs/paper_budget/qdislib/cuts")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--time-limit", type=int, default=600, help="seconds per find_cut call")
    ap.add_argument("--pilot", action="store_true", help="width 13, seed 1234, SparseCut k=1")
    args = ap.parse_args()
    src, out = Path(args.circuits), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cmap = [tuple(e) for e in json.loads((src / "coupling_map.json").read_text())]
    files = sorted(src.glob("holdout3_*.json"))[:args.limit]
    for p in files:
        meta = json.loads(p.read_text())
        qc = qiskit.qasm2.loads((src / f"{meta['cid']}.qasm").read_text(),
                                custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        check_numbering(qc, meta["gates"])
        rows = []
        for mode, kw in plan(args.pilot):
            extra = dict(coupling_map=cmap, layout=meta["uncut_phys"]) if mode == "sparsecut" else {}
            res, err, sec = timed_call(qc, args.time_limit, **kw, **extra)
            rows.append({"mode": mode, **kw, "result": res, "error": err, "seconds": round(sec, 3)})
        (out / f"{meta['cid']}.json").write_text(json.dumps(
            {"cid": meta["cid"], "qdislib": qd.__version__, "calls": rows}, default=to_json))
        n_err = sum(r["error"] is not None for r in rows)
        print(f"{meta['cid']}: {len(rows)} calls, {sum(r['seconds'] for r in rows):.1f} s, "
              f"errors {n_err}, max call {max(r['seconds'] for r in rows):.1f} s")


if __name__ == "__main__":
    main()
