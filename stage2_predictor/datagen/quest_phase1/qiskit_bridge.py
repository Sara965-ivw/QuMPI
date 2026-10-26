"""Bridge to the production qiskit/Aer stack.

STATUS: NOT validated in the development sandbox (no qiskit there). Before any
dataset generation, run tests/test_phase1.py AND tests/test_bridge.py (below)
on the pinned environment — this is a pilot exit criterion. The scientific
core (contraction, frame decomposition, candidate enumeration) is backend-
independent and already machine-precision-validated; this module only swaps
the simulation engine and the calibration source.

Three responsibilities:
  1. Circuit conversion (quest_phase1.Circuit <-> qiskit.QuantumCircuit).
  2. Live calibration snapshots from IBM Runtime, archived as dated raw JSON
     (Roadmap SS2.1 reproducibility rule).
  3. Aer density-matrix execution (GPU-capable) as a drop-in replacement for
     simulators.density_matrix at scale.

Cross-validation requirement: on 5 small circuits, Aer-backed labels must
match the numpy-core labels to atol 1e-6 (test_bridge.py sketches this).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .circuits import Circuit

_QISKIT_GATES_1Q = {"h", "x", "y", "z", "s", "t", "id", "rx", "ry", "rz"}
_QISKIT_GATES_2Q = {"cx", "cz", "rzz"}


def to_qiskit(circ: Circuit):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(circ.n_qubits)
    for g in circ.gates:
        if g.name not in _QISKIT_GATES_1Q | _QISKIT_GATES_2Q:
            raise ValueError(f"gate {g.name} has no qiskit mapping")
        getattr(qc, g.name)(*g.params, *g.qubits)
    return qc


def from_qiskit(qc) -> Circuit:
    circ = Circuit(qc.num_qubits)
    for inst in qc.data:
        name = inst.operation.name
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        params = tuple(float(p) for p in inst.operation.params)
        if name == "u":  # decompose if needed later; reject for now
            raise ValueError("decompose 'u' gates before conversion")
        circ.add(name, qubits if len(qubits) > 1 else qubits[0], *params)
    return circ


def fetch_snapshot(backend_name: str = "ibm_brisbane",
                   archive_dir: str = "calibration_snapshots") -> dict:
    """Pull live calibration and archive the dated raw JSON. Requires saved
    IBM Runtime credentials. Returns a snapshot dict in the schema of
    snapshot_fixture.json (plus the raw properties for the archive)."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    props = backend.properties()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    qubits = []
    for q in range(backend.num_qubits):
        qubits.append({
            "t1_us": props.t1(q) * 1e6,
            "t2_us": props.t2(q) * 1e6,
            "readout_p01": props.qubit_property(q, "prob_meas1_prep0")[0],
            "readout_p10": props.qubit_property(q, "prob_meas0_prep1")[0],
            "error_1q": _one_q_error(props, q),
        })
    snap = {
        "backend": backend_name,
        "calibration_date": stamp,
        "qubits": qubits,
        "errors_2q": _two_q_errors(props),
        "default_error_1q": 2e-4,
        "default_error_2q": 7e-3,
        "duration_1q_ns": 60.0,
        "duration_2q_ns": 500.0,
    }
    out = Path(archive_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{backend_name}_{stamp}.json").write_text(json.dumps(snap, indent=1))
    (out / f"{backend_name}_{stamp}_raw.json").write_text(json.dumps(props.to_dict(),
                                                                     default=str))
    return snap


def _one_q_error(props, q):
    for name in ("sx", "x", "rz"):
        try:
            return props.gate_error(name, [q])
        except Exception:
            continue
    return None  # loader will reject: better to fail loudly than default silently


def _two_q_errors(props) -> dict:
    errors = {}
    for gate in props.gates:
        if len(gate.qubits) == 2:
            try:
                err = props.gate_error(gate.gate, gate.qubits)
            except Exception:
                continue
            key = "_".join(map(str, sorted(gate.qubits)))
            errors[key] = err
    return errors


def _noise_model(noise_spec):
    """Aer NoiseModel equivalent to the numpy core's NoiseSpec.

    Composition matches the numpy engine: ideal gate -> depolarizing ->
    thermal relaxation (QuantumError.compose is left-to-right).

    Deliberately EXCLUDES readout error: the combiner applies readout as a
    classical channel after combination (identically for both engines), so
    putting it in the Aer model as well would double-count it.

    Two conventions the cross-validation (test_bridge.py) is designed to
    catch if wrong: (a) expand vs tensor ordering of the two-qubit
    relaxation channels — the fixture's asymmetric T1/T2 values make a
    swap detectable; (b) Aer's depolarizing_error parameter convention vs
    the numpy core's (both intend E(rho) = (1-p) rho + p I/2^n)."""
    from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                                  thermal_relaxation_error)
    nm = NoiseModel()
    if noise_spec is None:
        return nm
    for q, qn in noise_spec.qubits.items():
        relax = thermal_relaxation_error(qn.t1_us * 1e-6, qn.t2_us * 1e-6,
                                         noise_spec.duration_1q_ns * 1e-9)
        err1 = depolarizing_error(noise_spec.gate_error_1q.get(q, 0.0), 1)
        nm.add_quantum_error(err1.compose(relax),
                             ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "id"],
                             [q])
    pair_keys = set()
    for q1 in noise_spec.qubits:
        for q2 in noise_spec.qubits:
            if q1 < q2:
                pair_keys.add((q1, q2))
    for q1, q2 in pair_keys:
        p2 = noise_spec.gate_error_2q.get(frozenset((q1, q2)))
        r = [thermal_relaxation_error(noise_spec.qubits[q].t1_us * 1e-6,
                                      noise_spec.qubits[q].t2_us * 1e-6,
                                      noise_spec.duration_2q_ns * 1e-9)
             for q in (q1, q2)]
        # error qubit 0 must be the instruction's FIRST qubit (q1):
        relax2 = r[0].expand(r[1])
        for qubits in ((q1, q2), (q2, q1)):
            rr = relax2 if qubits[0] == q1 else r[1].expand(r[0])
            from qiskit_aer.noise import depolarizing_error as _de
            nm.add_quantum_error(_de(p2, 2).compose(rr),
                                 ["cx", "cz", "rzz"], list(qubits))
    return nm


def aer_density_matrix(circ: Circuit, noise_spec, init_psi=None, device="CPU"):
    """Aer drop-in for simulators.density_matrix; returns the density matrix
    in the NUMPY CORE'S tensor convention (shape (2,)*(2n), ket axes 0..n-1,
    axis q <-> qubit q, C-order/big-endian flattening).

    Endianness: qiskit is little-endian (qubit 0 = least significant bit);
    the numpy core's C-order flatten makes axis 0 most significant. Both the
    input state and the output matrix are therefore explicitly converted by
    reversing qubit axes. Runs WITHOUT transpilation: basis rewriting would
    rename gates and silently detach the noise model from them."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    n = circ.n_qubits
    qc = QuantumCircuit(n)
    if init_psi is not None:
        psi_le = np.transpose(np.asarray(init_psi, dtype=complex).reshape((2,) * n),
                              axes=list(reversed(range(n)))).reshape(-1)
        # Ideal (noise-free) preparation, matching the numpy engine. Aer's
        # density_matrix method rejects `initialize`; SetDensityMatrix is a
        # simulator instruction, so the noise model never attaches to it.
        from qiskit_aer.library import SetDensityMatrix
        qc.append(SetDensityMatrix(np.outer(psi_le, psi_le.conj())), list(range(n)))
    qc = qc.compose(to_qiskit(circ))
    qc.save_density_matrix()
    sim = AerSimulator(method="density_matrix",
                       noise_model=_noise_model(noise_spec), device=device)
    result = sim.run(qc).result()
    dm = np.asarray(result.data()["density_matrix"]).reshape((2,) * (2 * n))
    perm = list(reversed(range(n))) + [n + a for a in reversed(range(n))]
    return np.transpose(dm, perm)
