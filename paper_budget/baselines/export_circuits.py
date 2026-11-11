"""Export the holdout3 comparison circuits for Qdislib (thesis environment).

The circuits are the 143 of the find_cuts comparison (holdout3_selection.json),
so both baselines are compared on the same circuits. Per circuit:
  <cid>.qasm   the circuit, gate for gate (findcuts_map.to_qiskit)
  <cid>.json   gate list (for the numbering check), n_qubits, uncut layout (SparseCut)
plus coupling_map.json of the device, for Qdislib's hardware-aware mode.

Holdout4 (11 Nov 2026, notes/PREDICTIONS_compare_holdout4.md): --split holdout4 exports the
156 circuits of the pre-registered analysis set (holdout4_analysis_set.json, written by
paper_budget.holdout4_list) to runs/paper_budget/qdislib_holdout4/circuits. The default
(--split holdout3) is unchanged.
"""
import argparse
import json
import os
import sys

import qiskit.qasm2

from paper_budget.config import ROOT, DATA_V2, RUNS, DEVICE

sys.path.insert(0, str(ROOT / "stage2_predictor"))
sys.path.insert(0, str(ROOT / "stage2_predictor" / "analysis"))
from quest_phase2.data import rebuild_circuit  # noqa: E402  (frozen thesis code)
from findcuts_map import to_qiskit  # noqa: E402  (same conversion as for find_cuts)

SELECTION = {"holdout3": "holdout3_selection.json", "holdout4": "holdout4_analysis_set.json"}


def qdislib_dir(split: str):
    """Run folder of the Qdislib baseline: October layout for holdout3, a new folder otherwise."""
    return RUNS / ("qdislib" if split == "holdout3" else f"qdislib_{split}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="holdout3", choices=tuple(SELECTION))
    ap.add_argument("--limit", type=int, default=None, help="first N circuits only (pilot)")
    args = ap.parse_args()
    assert os.environ.get("QUEST_UNBLIND") == args.split, f"set QUEST_UNBLIND={args.split}"
    out = qdislib_dir(args.split) / "circuits"
    out.mkdir(parents=True, exist_ok=True)

    from quest_phase1.labeling_v2 import make_device
    ctx = make_device(DEVICE, seed=1234)
    edges = [list(e) for e in ctx.backend.coupling_map.get_edges()]
    (out / "coupling_map.json").write_text(json.dumps(edges))

    chosen = sorted(json.loads((DATA_V2 / SELECTION[args.split]).read_text())["circuits"])
    if args.limit:
        chosen = chosen[:args.limit]
    for cid in chosen:
        d = json.loads((DATA_V2 / args.split / f"{cid}.json").read_text())
        circ = rebuild_circuit(d["circuit_meta"])
        (out / f"{cid}.qasm").write_text(qiskit.qasm2.dumps(to_qiskit(circ)))
        (out / f"{cid}.json").write_text(json.dumps({
            "cid": cid, "n_qubits": circ.n_qubits,
            "gates": [[g.name, list(g.qubits)] for g in circ.gates],
            "uncut_phys": d["uncut_phys"]}))
    print(f"exported {len(chosen)} circuits and a {len(edges)}-edge coupling map to {out}")


if __name__ == "__main__":
    main()
