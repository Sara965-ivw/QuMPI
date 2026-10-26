"""Phase 0 exit test: full pipeline, no AI.
Stage 1: circuit -> partition          (manual labels = Stage 2/3 placeholder)
Stage 4: subexperiments on MPI ranks   (with barrier instrumentation)
Stage 5: QPD reconstruction vs exact   (first PST-style validation)
Run:  mpiexec -n 2 python stage5_reconstruct/smoke_test.py
"""
from mpi4py import MPI
import time, json
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_addon_cutting import (
    partition_problem,
    generate_cutting_experiments,
    reconstruct_expectation_values,
)
from qiskit_aer.primitives import SamplerV2

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
SHOTS = 8192

# ---- Stage 1: the circuit and its exact ground truth ----
qc = QuantumCircuit(6)
qc.h(0)
for i in range(5):
    qc.cx(i, i + 1)                      # 6-qubit GHZ; cx(2,3) crosses the cut

observable = SparsePauliOp(["ZZZZZZ"])
exact = Statevector(qc).expectation_value(observable).real

# ---- Stage 2/3 placeholder: manual partition, one wire boundary ----
partitioned = partition_problem(
    circuit=qc, partition_labels="AAABBB", observables=observable.paulis
)
subexperiments, coefficients = generate_cutting_experiments(
    circuits=partitioned.subcircuits,
    observables=partitioned.subobservables,
    num_samples=np.inf,                  # full QPD: no sampling of terms,
)                                        # residual error is shot noise only

labels = sorted(subexperiments.keys())   # ['A', 'B']
my_label = labels[rank]

# ---- Stage 4: distributed execution with barrier instrumentation ----
t0 = time.perf_counter()
sampler = SamplerV2()
job = sampler.run(subexperiments[my_label], shots=SHOTS)
result = job.result()
t_before = time.perf_counter()
comm.Barrier()
t_after = time.perf_counter()

timing = {
    "rank": rank,
    "partition": my_label,
    "n_subexperiments": len(subexperiments[my_label]),
    "sim_time": round(t_before - t0, 4),
    "barrier_time": round(t_after - t_before, 4),
}

gathered = comm.gather((my_label, result, timing), root=0)

# ---- Stage 5: reconstruction and validation (rank 0) ----
if rank == 0:
    results = {label: res for label, res, _ in gathered}
    timings = [t for _, _, t in gathered]

    reconstructed = reconstruct_expectation_values(
        results, coefficients, partitioned.subobservables
    )[0]

    print(json.dumps(timings, indent=2))
    print(f"\nExact  <ZZZZZZ> = {exact:+.4f}")
    print(f"Recon. <ZZZZZZ> = {reconstructed:+.4f}")
    print(f"Abs. error      = {abs(reconstructed - exact):.4f}")
    print("PASS" if abs(reconstructed - exact) < 0.05 else "FAIL")
