"""Card A, step 2: label the cut qiskit-addon-cutting find_cuts selects, for every
holdout3 circuit (decisions A1-A7, 22-23 Oct 2026; exploratory, see A1).

Per circuit:
  1. findcuts_map.find_feasible: find_cuts at qubits_per_subcircuit = 13 (A3); the width
     is lowered and find_cuts rerun only while a single subcircuit is too wide after
     routing (A5, corrected by fix A);
  2. the cut is labelled with the same device, noise model and transpiler seed as every
     other label (A2), each subcircuit simulated on its own as it would be executed:
       no cut needed  -> quest_phase1.nocut          (A4)
       gate cuts only -> quest_phase1.multipiece     (fix B, A7)
       with wire cut  -> quest_phase1.mixed_cuts     (two subcircuits = two sides)
     The noiseless oracle is checked on every new label. A labelled candidate of the
     circuit is reused only when it is the same execution: the same cut with exactly
     two subcircuits (tests_phase2/test_multipiece.py shows the two labellers then agree
     to 1e-10).
  3. Fix C: every circuit runs in its own process with a time limit; a circuit that
     exceeds it is recorded (<cid>.timeout.txt) and the run continues. Errors are
     recorded in <cid>.error.txt. A rerun retries only unfinished circuits.
Only counts and timing are printed; fidelities are read by findcuts_compare.py.

Holdout4 (11 Nov 2026, notes/PREDICTIONS_compare_holdout4.md): --split holdout4 --corpus <dir>
--selection <list>. With the defaults (--split holdout3, <split>_selection.json) the same
circuits are selected and labelled as in October (config.json records a few extra fields);
the labelling itself (label_one) is unchanged. holdout4 requires PYTHONHASHSEED=0, because the
multipiece labeller breaks layout ties by hash order (notes/RESULTS_findcuts_relabel.md).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "datagen", HERE):
    sys.path.insert(0, str(p))

LABEL_KEYS = ("cut_type", "g", "w", "k", "home_a", "wire_cuts", "gate_cut_indices", "pieces",
              "piece_widths", "width_a", "width_b", "log2_sampling_overhead", "gamma2",
              "pst_hellinger", "tv_distance", "obs_mu", "oracle_tvd", "oracle_obs_maxerr", "phys_a",
              "phys_b", "piece_phys", "routing_overhead_2q", "norm_residue", "negative_mass",
              "label_scheme", "label_wall_s")


def label_one(cid: str, h3_dir: Path, out_dir: Path) -> dict:
    import findcuts_map as F
    from quest_phase1.labeling_v2 import MappedWidthExceeded, make_device
    from quest_phase1.mixed_cuts import label_plan, plan_cut
    from quest_phase1.multipiece import label_multipiece
    from quest_phase1.nocut import label_nocut
    from quest_phase1.observables import basket, expectations
    from quest_phase1.simulators import probabilities
    from quest_phase2.data import rebuild_circuit
    t0 = time.time()
    c = json.loads((h3_dir / f"{cid}.json").read_text())
    meta = c["circuit_meta"]
    circ = rebuild_circuit(meta)
    ctx = make_device("fake_brisbane", seed=1234)
    p_ideal = probabilities(circ)
    spec = [tuple(x) for x in c["obs_spec"]] if c.get("obs_spec") else basket(circ)
    obs_ideal = c.get("obs_ideal") or expectations(p_ideal, spec)
    start, previous = F.QPU_WIDTH, []
    while True:
        res = F.find_feasible(ctx, circ, start=start)
        m = res["mapping"]
        row = {"cid": cid, "family": meta["family"], "n_qubits": circ.n_qubits, "status": res["status"],
               "kind": res["kind"], "width_used": res["width"], "attempts": previous + res["attempts"],
               "mapping": m.as_dict() if m is not None else None, "reused_candidate_label": False,
               "label": None}
        if res["status"] != "ok":
            break
        reusable = (m.n_fragments == 2 and m.representable and
                    [r for r in c["records"] if F.matches_candidate(m, r, circ.n_qubits)])
        try:
            if reusable:
                rec, row["reused_candidate_label"] = reusable[0], True
            elif res["kind"] == "nocut":
                rec = label_nocut(ctx, circ, p_ideal=p_ideal, obs_spec=spec, obs_ideal=obs_ideal)
            elif res["kind"] == "multipiece":
                rec = label_multipiece(ctx, circ, m.gate_cut_indices, p_ideal=p_ideal,
                                       obs_spec=spec, obs_ideal=obs_ideal)
            else:
                rec = label_plan(ctx, circ, plan_cut(circ, F.as_mixed_cut(circ, m)), p_ideal,
                                 c.get("f_uncut"), check_oracle=True, obs_spec=spec, obs_ideal=obs_ideal)
        except MappedWidthExceeded as e:
            # safety net: the feasibility check mirrors the labellers' mapping, so this
            # should not occur; if it does, A5 applies to the real labeller's verdict
            previous = row["attempts"] + [{"width": res["width"], "label_failed": str(e)}]
            start = res["width"] - 1
            if start < 2:
                row.update(status="infeasible", width_used=None, attempts=previous, mapping=None)
                break
            continue
        row["label"] = {k: rec[k] for k in LABEL_KEYS if k in rec}
        break
    row["wall_s"] = round(time.time() - t0, 1)
    tmp = out_dir / f"{cid}.tmp"
    tmp.write_text(json.dumps(row, indent=1))
    tmp.replace(out_dir / f"{cid}.json")
    return row


def worker(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", required=True)
    ap.add_argument("--holdout3", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    try:
        row = label_one(a.one, Path(a.holdout3), out)
        print(json.dumps({"status": row["status"], "kind": row["kind"], "width": row["width_used"],
                          "reused": row["reused_candidate_label"], "wall_s": row["wall_s"]}))
    except Exception:
        (out / f"{a.one}.error.txt").write_text(traceback.format_exc())
        raise


def main():
    if "--one" in sys.argv:
        return worker(sys.argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout3", "--corpus", dest="holdout3", required=True,
                    help="corpus directory of the split (corpus_v2/holdout3 or corpus_v2/holdout4)")
    ap.add_argument("--split", default="holdout3", choices=("holdout3", "holdout4"))
    ap.add_argument("--selection", default=None,
                    help="JSON with a 'circuits' list (default: <corpus parent>/<split>_selection.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--time-limit", type=int, default=1800, help="seconds per circuit (fix C)")
    ap.add_argument("--only", default=None, help="comma-separated circuit ids (testing)")
    a = ap.parse_args()
    if os.environ.get("QUEST_UNBLIND") != a.split:
        raise SystemExit(f"{a.split} is guarded: set QUEST_UNBLIND={a.split} (exploratory)")
    if a.split != "holdout3" and os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("export PYTHONHASHSEED=0 first (deterministic multipiece layouts)")
    h3 = Path(a.holdout3)
    assert h3.name == a.split, f"--corpus {h3} does not match --split {a.split}"
    sel = Path(a.selection) if a.selection else h3.parent / f"{a.split}_selection.json"
    chosen = sorted(json.loads(sel.read_text())["circuits"])
    if a.only:
        chosen = [c for c in chosen if c in set(a.only.split(","))]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "decisions": "A1 (d) exploratory on holdout3; A2 (a); A3 (a) width 13, defaults, seed 1234; "
                     "A4 (a) no-cut piecewise; A5 (a) with fix A: lower width only for a too-wide "
                     "subcircuit; A7 (b) fix B: each subcircuit simulated on its own; fix C: time limit",
        "split": a.split, "selection": str(sel), "n_circuits": len(chosen),
        "time_limit_s": a.time_limit, "workers": a.workers,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "git_commit": os.environ.get("QUEST_COMMIT", "unknown")}, indent=1))
    for cid in chosen:                    # a circuit that already exceeded the limit is not retried
        if (out / f"{cid}.timeout.txt").exists() and not (out / f"{cid}.json").exists():
            meta = json.loads((h3 / f"{cid}.json").read_text())["circuit_meta"]
            (out / f"{cid}.json").write_text(json.dumps({
                "cid": cid, "family": meta["family"], "n_qubits": meta["n_qubits"], "status": "timeout",
                "kind": None, "width_used": None, "attempts": [], "mapping": None,
                "reused_candidate_label": False, "label": None,
                "note": (out / f"{cid}.timeout.txt").read_text().strip()}, indent=1))
            print(f"{cid}: exceeded the time limit in an earlier run; recorded as excluded", flush=True)
    todo = [cid for cid in chosen if not (out / f"{cid}.json").exists()]
    for cid in todo:                      # a rerun retries circuits that ended in an error
        (out / f"{cid}.error.txt").unlink(missing_ok=True)
    print(f"{len(chosen) - len(todo)} already labelled, {len(todo)} to do", flush=True)
    t0, running, k = time.time(), {}, 0
    queue = list(todo)
    while queue or running:
        while queue and len(running) < a.workers:
            cid = queue.pop(0)
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--one", cid,
                   "--holdout3", str(h3), "--out", str(out)]
            running[cid] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                             text=True), time.time())
        time.sleep(1)
        for cid, (proc, started) in list(running.items()):
            elapsed = time.time() - started
            if proc.poll() is None:
                if elapsed > a.time_limit:
                    proc.kill(); proc.wait()
                    (out / f"{cid}.timeout.txt").write_text(f"killed after {a.time_limit} s\n")
                    # recorded as a result, so the comparison can run and lists it as excluded;
                    # delete this file to retry the circuit with a longer --time-limit
                    meta = json.loads((h3 / f"{cid}.json").read_text())["circuit_meta"]
                    (out / f"{cid}.json").write_text(json.dumps({
                        "cid": cid, "family": meta["family"], "n_qubits": meta["n_qubits"],
                        "status": "timeout", "kind": None, "width_used": None, "attempts": [],
                        "mapping": None, "reused_candidate_label": False, "label": None,
                        "wall_s": round(elapsed, 1), "time_limit_s": a.time_limit}, indent=1))
                    msg = f"TIMEOUT after {a.time_limit} s (recorded as excluded, run continues)"
                else:
                    continue
            else:
                lines = (proc.stdout.read() or "").strip().splitlines()
                msg = lines[-1] if proc.returncode == 0 and lines else f"ERROR (see {cid}.error.txt)"
            del running[cid]
            k += 1
            print(f"[{k}/{len(todo)}] {cid}: {msg} ({elapsed:.0f} s, elapsed {(time.time() - t0) / 60:.1f} min)",
                  flush=True)
    rows = [json.loads(p.read_text()) for p in sorted(out.glob(f"{a.split}_*.json"))]
    cnt = Counter((r["status"], r["kind"], r["reused_candidate_label"]) for r in rows)
    print("\nstatus, kind, reused candidate label -> circuits")
    for key, v in sorted(cnt.items(), key=str):
        print(f"  {key}: {v}")
    widths = Counter(r["width_used"] for r in rows)
    print("width used -> circuits:", dict(sorted(widths.items(), key=lambda x: (x[0] is None, x[0]))))
    errors = sorted(p.name for p in out.glob("*.error.txt"))
    timeouts = sorted(r["cid"] for r in rows if r["status"] == "timeout")
    if timeouts:
        print(f"{len(timeouts)} circuits exceeded the time limit and are excluded from the comparison: {timeouts}")
    if errors:
        print(f"{len(errors)} circuits ended in an error (rerun retries them): {errors}")
    print("LABELLING FINISHED" if len(rows) == len(chosen) else "LABELLING INCOMPLETE: rerun to resume")


if __name__ == "__main__":
    main()
