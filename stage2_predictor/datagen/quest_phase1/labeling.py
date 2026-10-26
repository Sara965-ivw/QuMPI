"""Pilot labelling loop (Roadmap SS4.3 with amended D1) + additivity probe (F2).

Outputs one JSON record per (circuit, candidate) with: family metadata, cut
configuration, PST label (Hellinger fidelity, primary), TV distance
(secondary), node features (both aggregation channels), snapshot provenance,
and wall time. The noiseless-oracle invariant is checked on every candidate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .candidates import enumerate_candidates, validate_candidate
from .combiner import label_candidate
from .noise import build_noise_spec, gate_node_features, load_snapshot


def label_circuit(circ, meta, snap, k_max=2, check_oracle=True):
    noise = build_noise_spec(snap, circ.n_qubits)
    feats = gate_node_features(circ, snap)
    records = []
    for cand in enumerate_candidates(circ, k_max=k_max):
        validate_candidate(circ, cand)
        t0 = time.perf_counter()
        rec = label_candidate(circ, cand, noise, check_oracle=check_oracle)
        rec.update({"circuit_meta": meta, "label_wall_s": time.perf_counter() - t0,
                    "snapshot_backend": snap["backend"],
                    "snapshot_date": snap["calibration_date"]})
        records.append(rec)
    return records, feats


def run_pilot(circuits_with_meta, snapshot_path, out_path, k_max=2):
    snap = load_snapshot(snapshot_path)
    all_records, all_feats = [], []
    for circ, meta in circuits_with_meta:
        recs, feats = label_circuit(circ, meta, snap, k_max=k_max)
        all_records.extend(recs)
        all_feats.append({"circuit_meta": meta, "node_features": feats})
    labels = np.array([r["pst_hellinger"] for r in all_records])
    summary = {
        "n_circuits": len(circuits_with_meta),
        "n_labelled_candidates": len(all_records),
        "label_mean": float(labels.mean()), "label_std": float(labels.std()),
        "label_min": float(labels.min()), "label_max": float(labels.max()),
        # F3 exit criterion: histogram to detect compression near 1.
        "label_histogram": {
            "bin_edges": [round(x, 3) for x in np.linspace(labels.min(), 1.0, 11)],
            "counts": np.histogram(labels, bins=np.linspace(labels.min(), 1.0, 11))[0].tolist(),
        },
        "total_label_wall_s": float(sum(r["label_wall_s"] for r in all_records)),
        "snapshot": {"backend": snap["backend"], "date": snap["calibration_date"],
                     "path": str(snapshot_path)},
    }
    out = {"summary": summary, "records": all_records, "features": all_feats}
    Path(out_path).write_text(json.dumps(out, indent=1))
    return out


def additivity_probe(records):
    """F2: for k=2 candidates whose part_a strictly contains a k=1 candidate's
    part_a on the same circuit, compare labels. Returns paired rows; analysis
    (does PST(A2) track f(PST(A1), ...)?) is reported in the datasheet."""
    by_circuit: dict = {}
    for r in records:
        key = json.dumps(r["circuit_meta"], sort_keys=True)
        by_circuit.setdefault(key, []).append(r)
    pairs = []
    for recs in by_circuit.values():
        k1 = [r for r in recs if r["k"] == 1]
        k2 = [r for r in recs if r["k"] == 2]
        for r2 in k2:
            for r1 in k1:
                if r1["n_gates_a"] < r2["n_gates_a"]:
                    pairs.append({"pst_k1": r1["pst_hellinger"],
                                  "pst_k2": r2["pst_hellinger"],
                                  "circuit_meta": r1["circuit_meta"]})
    return pairs
