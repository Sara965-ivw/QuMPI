"""Export the holdout3 comparison circuits for Qdislib (thesis environment).

The circuits are the 143 of the find_cuts comparison (holdout3_selection.json),
so both baselines are compared on the same circuits. Per circuit:
  <cid>.qasm   the circuit, gate for gate (findcuts_map.to_qiskit)
  <cid>.json   gate list (for the numbering check), n_qubits, uncut layout (SparseCut)
plus coupling_map.json of the device, for Qdislib's hardware-aware mode.
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

OUT = RUNS / "qdislib" / "circuits"


def main():
    assert os.environ.get("QUEST_UNBLIND") == "holdout3", "set QUEST_UNBLIND=holdout3"
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N circuits only (pilot)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from quest_phase1.labeling_v2 import make_device
    ctx = make_device(DEVICE, seed=1234)
    edges = [list(e) for e in ctx.backend.coupling_map.get_edges()]
    (OUT / "coupling_map.json").write_text(json.dumps(edges))

    chosen = sorted(json.loads((DATA_V2 / "holdout3_selection.json").read_text())["circuits"])
    if args.limit:
        chosen = chosen[:args.limit]
    for cid in chosen:
        d = json.loads((DATA_V2 / "holdout3" / f"{cid}.json").read_text())
        circ = rebuild_circuit(d["circuit_meta"])
        (OUT / f"{cid}.qasm").write_text(qiskit.qasm2.dumps(to_qiskit(circ)))
        (OUT / f"{cid}.json").write_text(json.dumps({
            "cid": cid, "n_qubits": circ.n_qubits,
            "gates": [[g.name, list(g.qubits)] for g in circ.gates],
            "uncut_phys": d["uncut_phys"]}))
    print(f"exported {len(chosen)} circuits and a {len(edges)}-edge coupling map to {OUT}")


if __name__ == "__main__":
    main()
