from mpi4py import MPI
import time, json, sys

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

imbalance = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

t0 = time.perf_counter()
time.sleep(0.5 + imbalance * rank)      # synthetic "simulation" work
t_before_barrier = time.perf_counter()
comm.Barrier()
t_after_barrier = time.perf_counter()

record = {
    "rank": rank,
    "sim_time": round(t_before_barrier - t0, 4),
    "barrier_time": round(t_after_barrier - t_before_barrier, 4),
}
all_records = comm.gather(record, root=0)
if rank == 0:
    print(json.dumps(all_records, indent=2))
