"""Card A, step 1: dry run of qiskit-addon-cutting find_cuts on the holdout3 analysis set.

No fidelity, label or model score is read or computed here. For each of the 143
pre-registered holdout3 circuits the script records:
  * the cut find_cuts selects (decision A3: qubits_per_subcircuit = 13, defaults otherwise, seed 1234);
  * whether the labeller can represent it (two sides, wire cuts A -> B once);
  * whether both fragments fit the labeller's 13-qubit ceiling AFTER routing on
    fake_brisbane (the same feasibility filter applied to our own candidates);
  * whether the cut is already one of the circuit's labelled candidates (then its
    label exists and nothing new has to be simulated);
  * whether find_cuts reports that no cut is needed (circuit already separable).
The counts decide how much labeller extension decision A2 requires.

holdout3 was opened once on 16 Oct 2026 (PREREG_holdout3.md); this comparison is
exploratory (decision A1) and reads circuit structure only.
"""
import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "datagen")); sys.path.insert(0, str(HERE))

from qiskit.circuit import Parameter                                       # noqa: E402

import findcuts_map as F                                                   # noqa: E402
from quest_phase1.labeling_v2 import MappedWidthExceeded, make_device      # noqa: E402
from quest_phase1.mixed_cuts import _RZZ_TERMS, build_fragments, map_fragment, plan_cut  # noqa: E402
from quest_phase2.data import rebuild_circuit                              # noqa: E402


def routed_fits(ctx, circ, plan):
    """Map both fragments exactly as the labeller does (first QPD term); report widths."""
    ops = {i: (_RZZ_TERMS[0][1], _RZZ_TERMS[0][2]) for i in plan.gate_cuts}
    params = {(i, sd): Parameter(f"g{i}{sd}") for i in plan.gate_cuts for sd in "AB"}
    fa, fb = build_fragments(circ, plan, ops, params)
    out = []
    for n_loc, fr in ((plan.width_a, fa), (plan.width_b, fb)):
        try:
            map_fragment(ctx, n_loc, fr)
            out.append("ok")
        except MappedWidthExceeded as e:
            out.append(str(e).split(" qubits")[0])
    return all(o == "ok" for o in out), out


def matches_candidate(m, rec, n):
    """Same bipartition (or its mirror for pure gate cuts) and the same cuts."""
    if sorted(rec["gate_cut_indices"]) != sorted(m.gate_cut_indices):
        return False
    if sorted(map(tuple, rec["wire_cuts"])) != sorted(map(tuple, m.wire_cuts)):
        return False
    a = set(rec["home_a"])
    return a == set(m.home_a) or (not m.wire_cuts and a == set(range(n)) - set(m.home_a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout3", required=True, help="corpus_v2/holdout3 directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.environ.get("QUEST_UNBLIND") != "holdout3":
        raise SystemExit("holdout3 is guarded: set QUEST_UNBLIND=holdout3 (exploratory, decision A1)")
    h3 = Path(a.holdout3)
    chosen = json.loads((h3.parent / "holdout3_selection.json").read_text())["circuits"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ctx = make_device("fake_brisbane", seed=1234)
    import qiskit_addon_cutting
    rows, t0 = [], time.time()
    for k, cid in enumerate(sorted(chosen), 1):
        c = json.loads((h3 / f"{cid}.json").read_text())
        meta = c["circuit_meta"]
        circ = rebuild_circuit(meta)
        t = time.time()
        qc, md = F.run_find_cuts(circ)
        m = F.map_result(circ, qc, md)
        row = {"cid": cid, "family": meta["family"], "n_qubits": circ.n_qubits,
               "findcuts_s": round(time.time() - t, 2), **m.as_dict(),
               "routed_fits": None, "routed_widths": None, "in_candidates": False}
        if m.representable:
            plan = plan_cut(circ, F.as_mixed_cut(circ, m))
            row["routed_fits"], row["routed_widths"] = routed_fits(ctx, circ, plan)
            row["in_candidates"] = any(matches_candidate(m, r, circ.n_qubits) for r in c["records"])
        rows.append(row)
        print(f"[{k}/{len(chosen)}] {cid}: g={m.n_gate_cuts} w={m.n_wire_cuts} pieces={m.n_fragments} "
              f"{'NO CUT NEEDED' if m.no_cut_needed else m.reason} fits={row['routed_fits']} "
              f"in_candidates={row['in_candidates']}", flush=True)

    def bucket(r):
        if r["no_cut_needed"]:
            return "no cut needed (circuit separable into pieces <= 13 q)"
        if not r["representable"]:
            return "not representable: " + r["reason"]
        if not r["routed_fits"]:
            return "representable, routed fragment > 13 q"
        return "representable, fits, already a labelled candidate" if r["in_candidates"] \
            else "representable, fits, needs a new label"

    cnt = collections.Counter(bucket(r) for r in rows)
    by_cell = collections.defaultdict(collections.Counter)
    for r in rows:
        by_cell[(r["family"], r["n_qubits"])][bucket(r)] += 1
    summary = {"n": len(rows), "counts": dict(cnt),
               "by_cell": {f"{f}_{n}": dict(v) for (f, n), v in sorted(by_cell.items())},
               "findcuts_wire_cuts_total": sum(r["n_wire_cuts"] for r in rows),
               "overhead_models_agree": all(abs(r["findcuts_gamma"] - 2 ** r["labeller_log2_overhead"])
                                            < 1e-6 * r["findcuts_gamma"] for r in rows if r["representable"]),
               "qiskit_addon_cutting": getattr(qiskit_addon_cutting, "__version__", "unknown"),
               "settings": {"qubits_per_subcircuit": F.QPU_WIDTH, "seed": F.FINDCUTS_SEED,
                            "other_parameters": "defaults (max_gamma 1024, max_backjumps 10000, gate_lo, wire_lo)"},
               "git_commit": os.environ.get("QUEST_COMMIT", "unknown"),
               "minutes": round((time.time() - t0) / 60, 1)}
    (out / "dryrun_rows.json").write_text(json.dumps(rows, indent=1))
    (out / "dryrun_summary.json").write_text(json.dumps(summary, indent=1))
    print("\n=== find_cuts dry run on holdout3 (exploratory) ===")
    for k_, v in cnt.most_common():
        print(f"  {v:>4}  {k_}")
    print(f"  wire cuts chosen in total: {summary['findcuts_wire_cuts_total']};  "
          f"overhead models agree: {summary['overhead_models_agree']}")
    print("DRY RUN FINISHED")


if __name__ == "__main__":
    main()
