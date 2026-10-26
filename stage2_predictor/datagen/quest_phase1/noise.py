"""Calibration snapshot loading -> (node features, NoiseSpec).

Reproducibility rule (Roadmap SS2.1, Decision Record D3): every generation run
records the exact calibration snapshot it used — backend name, calibration
date, raw JSON — alongside the dataset. This module consumes such a snapshot;
snapshot_fixture.json ships as a schema example with brisbane-class magnitudes
(T1 ~ 200 us, T2 ~ 130 us, 2q error ~ 7e-3, readout ~ 1e-2). Live snapshots
come from qiskit_bridge.fetch_snapshot() on a machine with IBM credentials.

Feature policy (D3): BOTH aggregation channels (min and mean) are stored for
two-qubit gates so the Phase 2 ablation needs no dataset regeneration.
"""
from __future__ import annotations

import json
from pathlib import Path

from .circuits import Circuit
from .simulators import NoiseSpec, QubitNoise

REQUIRED_QUBIT_KEYS = ("t1_us", "t2_us", "readout_p01", "readout_p10")


def load_snapshot(path) -> dict:
    snap = json.loads(Path(path).read_text())
    for key in ("backend", "calibration_date", "qubits"):
        if key not in snap:
            raise ValueError(f"snapshot missing '{key}'")
    for q, props in enumerate(snap["qubits"]):
        for k in REQUIRED_QUBIT_KEYS:
            if props.get(k) is None:
                raise ValueError(f"qubit {q}: missing/None '{k}'")
    return snap


def build_noise_spec(snap: dict, n_qubits: int) -> NoiseSpec:
    if n_qubits > len(snap["qubits"]):
        raise ValueError("circuit wider than snapshot qubit count")
    spec = NoiseSpec(
        duration_1q_ns=snap.get("duration_1q_ns", 60.0),
        duration_2q_ns=snap.get("duration_2q_ns", 500.0),
    )
    for q in range(n_qubits):
        p = snap["qubits"][q]
        spec.qubits[q] = QubitNoise(p["t1_us"], p["t2_us"],
                                    p["readout_p01"], p["readout_p10"])
        spec.gate_error_1q[q] = p.get("error_1q", snap.get("default_error_1q", 2e-4))
    default_2q = snap.get("default_error_2q", 7e-3)
    pair_errors = {frozenset(map(int, k.split("_"))): v
                   for k, v in snap.get("errors_2q", {}).items()}
    # Any pair not in the snapshot gets the backend default (pre-mapping
    # assumption D3(iii): device-class aggregate statistics as fallback).
    spec.gate_error_2q = _DefaultDict(pair_errors, default_2q)
    return spec


class _DefaultDict(dict):
    def __init__(self, data, default):
        super().__init__(data)
        self._default = default

    def get(self, key, fallback=None):  # noqa: A003
        return super().get(key, self._default)


def gate_node_features(circ: Circuit, snap: dict) -> list:
    """Per-gate feature dicts with BOTH aggregation channels (D3).

    Raw qubit indices are intentionally NOT features (SS2.1-C index-encoding
    fix); they are carried as metadata for graph construction only.
    """
    spec = build_noise_spec(snap, circ.n_qubits)
    feats = []
    for g in circ.gates:
        t1s = [spec.qubits[q].t1_us for q in g.qubits]
        t2s = [spec.qubits[q].t2_us for q in g.qubits]
        err = (spec.gate_error_1q[g.qubits[0]] if g.n_qubits == 1
               else spec.gate_error_2q.get(frozenset(g.qubits)))
        row = {"gate_type": g.name,
               "t1_min": min(t1s), "t1_mean": sum(t1s) / len(t1s),
               "t2_min": min(t2s), "t2_mean": sum(t2s) / len(t2s),
               "gate_error": err,
               "_qubits_meta": list(g.qubits)}
        if any(v is None for v in row.values()):
            raise ValueError("None survived into features (regression rule)")
        feats.append(row)
    return feats
