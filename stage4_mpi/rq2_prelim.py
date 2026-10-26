"""RQ2 preliminary study: barrier-time inversion in real cut-circuit workloads.

Question
--------
Does barrier-time inversion (a rank waiting longer at the reconstruction
barrier than it computed, rho_r = w_r / t_r > 1) occur when real circuit-cutting
subexperiments are executed on MPI ranks, does it track partition imbalance,
and does a cost-aware assignment remove it?

Analytic expectation. With w_r = t_max - t_r,
    rho_r > 1  <=>  t_r < t_max / 2,
so inversion appears exactly when some rank receives less than half the work of
the slowest rank. Parallel efficiency E = 1 - sum_r w_r / (n t_max).

Workload. The tasks are the independent fragment simulations that the
labelling pipeline itself performs for one cut (mixed_cuts.reconstruct_plan):
one noisy density-matrix simulation per distinct fragment-A operator setting
and one per (fragment-B setting, wire-cut preparation frame). The task list is
checked against runs_a + runs_b stored in the corpus record.

Assignment policies (independent variable)
  fragment   ranks split evenly between fragments A and B, round-robin inside
             each group (one fragment per worker group; structural, cost-blind)
  count      equal number of tasks per rank, round-robin over all tasks
             (balances task COUNT, as METIS balances vertex count, not time)
  lpt        static longest-processing-time-first on a predicted cost
             c = 4^(simulated qubits) * operations (Graham 1969); the static
             assignment a load-balance term in the QUBO would produce
  dynamic    self-scheduling from a shared atomic counter, tasks in descending
             predicted cost (the strongest runtime baseline: if it removes
             inversion, a static balance term only matters where assignment
             must be fixed in advance, e.g. fragments pinned to devices)

Controls: one Aer thread per rank (max_parallel_threads=1, OMP_NUM_THREADS=1),
ranks <= physical cores, no other jobs, an untimed warm-up pass, policies run in
a seeded random order inside each repetition, transpilation outside the timed
region, time.perf_counter wall time.

Usage (from quest.sh rq2-prelim):
  mpirun -np 4 python rq2_prelim.py --corpus DIR --workload 0 --out OUT.jsonl
  python rq2_prelim.py --corpus DIR --list          # print the workload table
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage2_predictor" / "datagen"))
sys.path.insert(0, str(HERE.parent / "stage2_predictor"))

POLICIES = ("fragment", "count", "lpt", "dynamic")
FAMILIES = ("clustered_random", "twolocal", "qaoa_sbm")
WIDTHS = (12, 14)
PER_CELL = 2              # circuits per (family, width): 3 x 2 x 2 = 12 circuits, 24 workloads
REPEATS = 3
ORDER_SEED = 2026


# ------------------------------------------------------------------ workloads
def select_workloads(corpus: Path, widths=WIDTHS, per_cell=PER_CELL) -> list:
    """Deterministic, pre-declared selection from the TRAIN split (never a holdout):
    per (family, width) the first PER_CELL circuits by id with >= 2 candidates of
    different overhead; two cuts each: the minimum-overhead cut (first on ties)
    and the highest-fidelity cut (next best if they coincide)."""
    out = []
    files = sorted((corpus / "train").glob("*.json"))
    for fam in FAMILIES:
        for n in widths:
            chosen = 0
            for f in files:
                if f"_{fam}_n{n:02d}_" not in f.name or chosen == per_cell:
                    continue
                recs = json.loads(f.read_text()).get("records") or []
                if len({round(r["log2_sampling_overhead"], 9) for r in recs}) < 2:
                    continue
                chosen += 1
                i_mo = min(range(len(recs)), key=lambda i: (recs[i]["log2_sampling_overhead"], i))
                by_f = sorted(range(len(recs)), key=lambda i: -recs[i]["pst_hellinger"])
                i_bf = by_f[0] if by_f[0] != i_mo else by_f[1]
                for tag, i in (("min_overhead", i_mo), ("best_fidelity", i_bf)):
                    out.append({"workload": len(out), "file": f.name, "family": fam,
                                "n_qubits": n, "cut": tag, "record": i})
    return out


def build_tasks(ctx, corpus: Path, wl: dict) -> list:
    """Enumerate the fragment-simulation tasks of one cut, exactly as
    reconstruct_plan would execute them, without running them."""
    from quest_phase1.mixed_cuts import (MixedCut, _RZZ_TERMS, _SLOT_ANGLE, Mapped,
                                         _term_weights, build_fragments, map_fragment,
                                         plan_cut)
    from quest_phase2.data import rebuild_circuit
    from qiskit.circuit import Parameter

    d = json.loads((corpus / "train" / wl["file"]).read_text())
    rec = d["records"][wl["record"]]
    circ = rebuild_circuit(d["circuit_meta"])
    cut = MixedCut(frozenset(rec["home_a"]), tuple(tuple(x) for x in rec["wire_cuts"]),
                   circ.n_qubits)
    plan = plan_cut(circ, cut)
    assert plan is not None and list(plan.gate_cuts) == list(rec["gate_cut_indices"]), \
        "plan reconstruction does not match the corpus record"
    nk, nc, nr = len(plan.k_qubits), len(plan.c_qubits), len(plan.r_qubits)
    na, nb = nk + nc, nc + nr
    params = {(i, sd): Parameter(f"g{i}{sd}") for i in plan.gate_cuts for sd in "AB"}
    tmpl_a, tmpl_b, keys_a, keys_b = {}, {}, {}, {}
    layout_a = layout_b = None

    def bound(m, side, key):
        values = {params[(i, side)]: _SLOT_ANGLE[op]
                  for i, op in zip(plan.gate_cuts, key) if op != "M"}
        live = {k: v for k, v in values.items() if k in m.tqc.parameters}
        tqc = m.tqc.assign_parameters(live) if live else m.tqc
        return Mapped(tqc, m.initial, m.final, m.n_meas, m.n_2q_logical, m.n_2q_mapped, m.depth)

    weights = {i: _term_weights(math.pi / 2 if circ.gates[i].name in ("cx", "cz")
                                else float(circ.gates[i].params[0]))
               for i in plan.gate_cuts}
    for tau in itertools.product(range(6), repeat=plan.g):
        if math.prod(weights[i][t] for i, t in zip(plan.gate_cuts, tau)) == 0.0:
            continue                                   # skipped by reconstruct_plan too
        ops = {i: (_RZZ_TERMS[t][1], _RZZ_TERMS[t][2]) for i, t in zip(plan.gate_cuts, tau)}
        key_a = tuple(ops[i][0] for i in plan.gate_cuts)
        key_b = tuple(ops[i][1] for i in plan.gate_cuts)
        sa, sb = tuple(o == "M" for o in key_a), tuple(o == "M" for o in key_b)
        if sa not in tmpl_a or sb not in tmpl_b:
            fa, fb = build_fragments(circ, plan, ops, params)
        if sa not in tmpl_a:
            tmpl_a[sa] = map_fragment(ctx, na, fa, initial_layout=layout_a)
            layout_a = layout_a or tmpl_a[sa].initial
        if sb not in tmpl_b:
            tmpl_b[sb] = map_fragment(ctx, nb, fb, initial_layout=layout_b)
            layout_b = layout_b or tmpl_b[sb].initial
        keys_a.setdefault(key_a, bound(tmpl_a[sa], "A", key_a))
        keys_b.setdefault(key_b, bound(tmpl_b[sb], "B", key_b))

    tasks = []
    for key, m in keys_a.items():
        tasks.append({"side": "A", "m": m, "n_local": na, "prep": None,
                      "instr": tuple(range(nk, na))})
    for key, m in keys_b.items():
        for s_flat in range(4 ** nc):
            s = [(s_flat // 4 ** (nc - 1 - j)) % 4 for j in range(nc)]
            prep = {m.initial[j]: s[j] for j in range(nc) if s[j] != 0}
            tasks.append({"side": "B", "m": m, "n_local": nb, "prep": prep, "instr": ()})
    expected = rec["runs_a"] + rec["runs_b"]
    assert len(tasks) == expected, f"task count {len(tasks)} != runs_a + runs_b = {expected}"
    for t in tasks:
        m = t["m"]
        active = {m.tqc.find_bit(q).index for inst in m.tqc.data
                  if inst.operation.name not in ("barrier", "delay") for q in inst.qubits}
        n_sim = len(active) + m.n_meas
        t["cost"] = float(4.0 ** n_sim * max(1, len(m.tqc.data)))
        t["n_sim"] = n_sim
    return tasks


# ------------------------------------------------------------------ assignment
def assign(policy: str, tasks: list, n: int) -> list:
    """Static assignment: list of task-index lists, one per rank."""
    idx = list(range(len(tasks)))
    if policy == "count":
        return [idx[r::n] for r in range(n)]
    if policy == "fragment":
        a = [i for i in idx if tasks[i]["side"] == "A"]
        b = [i for i in idx if tasks[i]["side"] == "B"]
        na_r = max(1, n // 2)
        ranks_a, ranks_b = list(range(na_r)), list(range(na_r, n)) or [0]
        out = [[] for _ in range(n)]
        for j, i in enumerate(a):
            out[ranks_a[j % len(ranks_a)]].append(i)
        for j, i in enumerate(b):
            out[ranks_b[j % len(ranks_b)]].append(i)
        return out
    if policy == "lpt":
        out, load = [[] for _ in range(n)], [0.0] * n
        for i in sorted(idx, key=lambda i: -tasks[i]["cost"]):
            r = min(range(n), key=lambda r: (load[r], r))
            out[r].append(i)
            load[r] += tasks[i]["cost"]
        return out
    raise ValueError(policy)


# ------------------------------------------------------------------ execution
def run_task(ctx, t):
    from quest_phase1.mixed_cuts import simulate_mapped
    simulate_mapped(ctx, t["m"], t["n_local"], prep_phys=t["prep"], instrument_local=t["instr"])


def timed_run(comm, ctx, tasks, policy, win=None):
    rank, n = comm.Get_rank(), comm.Get_size()
    per_task = []
    comm.Barrier()                                   # common start
    t0 = time.perf_counter()
    if policy == "dynamic":
        from mpi4py import MPI
        order = sorted(range(len(tasks)), key=lambda i: -tasks[i]["cost"])
        one, got = np.array([1], dtype="i8"), np.array([0], dtype="i8")
        while True:
            win.Lock(0, MPI.LOCK_SHARED)
            win.Fetch_and_op(one, got, 0, 0, MPI.SUM)
            win.Unlock(0)
            k = int(got[0])
            if k >= len(order):
                break
            s = time.perf_counter(); run_task(ctx, tasks[order[k]])
            per_task.append((order[k], time.perf_counter() - s))
    else:
        for i in assign(policy, tasks, n)[rank]:
            s = time.perf_counter(); run_task(ctx, tasks[i])
            per_task.append((i, time.perf_counter() - s))
    t1 = time.perf_counter()
    comm.Barrier()
    t2 = time.perf_counter()
    return {"rank": rank, "sim_time": t1 - t0, "barrier_time": t2 - t1,
            "n_tasks": len(per_task), "task_times": per_task}


def make_ctx():
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from quest_phase1.labeling_v2 import make_device
    ctx = make_device("fake_brisbane", seed=1234)
    nm = NoiseModel.from_backend(ctx.backend, readout_error=False)
    ctx._sim = AerSimulator(method="density_matrix", noise_model=nm, max_parallel_threads=1)
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--workload", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--widths", default=",".join(map(str, WIDTHS)))   # testing only
    a = ap.parse_args()
    wls = select_workloads(a.corpus, tuple(int(x) for x in a.widths.split(",")))
    if a.list:
        for w in wls:
            print(json.dumps(w))
        return
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    wl = wls[a.workload]
    ctx = make_ctx()
    tasks = build_tasks(ctx, a.corpus, wl)            # identical on every rank (fixed seed)
    for t in tasks[rank::n][:1]:                      # untimed warm-up: Aer + noise-model init
        run_task(ctx, t)
    counter = np.zeros(1, dtype="i8")
    win = MPI.Win.Create(counter if rank == 0 else np.zeros(0, dtype="i8"), comm=comm)
    rng = np.random.default_rng(ORDER_SEED + 1000 * a.workload + n)
    rows = []
    for rep in range(a.repeats):
        for policy in rng.permutation(POLICIES):
            if rank == 0:                              # reset the shared task counter
                win.Lock(0, MPI.LOCK_EXCLUSIVE)
                win.Put(np.zeros(1, dtype="i8"), 0, 0)
                win.Unlock(0)
            comm.Barrier()
            res = timed_run(comm, ctx, tasks, str(policy), win)
            allres = comm.gather(res, root=0)
            if rank == 0:
                pred = [0.0] * n
                if policy != "dynamic":
                    for r, idx in enumerate(assign(str(policy), tasks, n)):
                        pred[r] = sum(tasks[i]["cost"] for i in idx)
                rows.append({**wl, "ranks": n, "policy": str(policy), "repeat": rep,
                             "n_tasks": len(tasks), "pred_load": pred,
                             "task_cost": [t["cost"] for t in tasks],
                             "task_side": [t["side"] for t in tasks],
                             "per_rank": allres})
    win.Free()
    if rank == 0:
        meta = {"host": platform.node(), "python": platform.python_version(),
                "omp": os.environ.get("OMP_NUM_THREADS"), "commit": os.environ.get("QUEST_COMMIT")}
        with open(a.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "env": meta}) + "\n")
        print(f"workload {a.workload} ({wl['family']} n={wl['n_qubits']} {wl['cut']}): "
              f"{len(tasks)} tasks, {n} ranks, {len(rows)} timed runs", flush=True)


if __name__ == "__main__":
    main()
