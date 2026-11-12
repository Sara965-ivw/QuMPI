"""Relabel the holdout4 analysis set on another device (notes/PREREG_transfer_sherbrooke.md).

Same circuits, same candidate cuts, same observables, same labeller; only the device changes.
For every circuit of data/corpus_v2/holdout4_analysis_set.json:
  - the circuit is rebuilt from its stored meta data (quest_phase2.data.rebuild_circuit, frozen);
  - its ideal observable values are recomputed and must equal the stored ones (1e-9): same circuit;
  - every candidate of the fake_brisbane file is rebuilt from its record (plan_cut), its structure
    (cut type, gate cuts, widths, sampling overhead) must equal the record, and it is labelled with
    the frozen label_plan on the new device; the oracle check is run on the same records as before;
  - a candidate whose routed fragment exceeds the 13-qubit simulation ceiling on the new device is
    dropped and counted (n_excluded_mapped_width), exactly as label_circuit_v3 does;
  - the uncut reference layout and the node features are recomputed for the new device.
Output: data/corpus_v2/holdout4_<device without "fake_">/<cid>.json in the format of label_circuit_v3,
and manifest_holdout4_<device>.json. Resumable; an existing output file is never overwritten.
Prints structure and timing only (number of candidates kept), never a label value.

  QUEST_UNBLIND=holdout4 PYTHONHASHSEED=0 python -m paper_budget.datagen.relabel_device --estimate
  QUEST_UNBLIND=holdout4 PYTHONHASHSEED=0 python -m paper_budget.datagen.relabel_device \\
      --device fake_sherbrooke --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from paper_budget.config import DATA_V2, ROOT

sys.path.insert(0, str(ROOT / "stage2_predictor" / "datagen"))
import corpus  # noqa: E402  (frozen thesis code: TRANSPILER_SEED, SETTINGS, _manifest)

SOURCE = "holdout4"
SELECTION = DATA_V2 / "holdout4_analysis_set.json"
STRUCT = ("cut_type", "g", "w", "width_a", "width_b", "gate_cut_indices", "home_a", "wire_cuts")


def out_split(device: str) -> str:
    return f"{SOURCE}_{device.removeprefix('fake_')}"


def rebuild_plan(circ, rec: dict):
    """The CutPlan of a stored record (the construction quest_phase2.data.build_graph uses)."""
    from quest_phase1.mixed_cuts import MixedCut, plan_cut
    plan = plan_cut(circ, MixedCut(frozenset(rec["home_a"]), tuple(tuple(x) for x in rec["wire_cuts"]),
                                   circ.n_qubits))
    if plan is None:
        raise ValueError("stored candidate is not a valid cut of the rebuilt circuit")
    got = {"cut_type": plan.cut_type, "g": plan.g, "w": plan.w, "width_a": plan.width_a,
           "width_b": plan.width_b, "gate_cut_indices": list(plan.gate_cuts),
           "home_a": sorted(plan.cut.home_a), "wire_cuts": [list(x) for x in plan.cut.wire_cuts]}
    for k in STRUCT:
        if got[k] != rec[k]:
            raise ValueError(f"rebuilt candidate differs from the record in {k}: {got[k]} vs {rec[k]}")
    if abs(plan.log2_overhead - rec["log2_sampling_overhead"]) > 1e-9:
        raise ValueError("rebuilt candidate has a different sampling overhead")
    return plan


def relabel_one(job: tuple) -> tuple:
    """Label one holdout4 circuit on `device` and write it. Returns (cid, status, wall s, kept, dropped)."""
    cid, device = job
    path = DATA_V2 / out_split(device) / f"{cid}.json"
    if path.exists():
        return cid, "skipped", 0.0, 0, 0
    out = relabel_circuit(DATA_V2 / SOURCE / f"{cid}.json", device)
    out.update(split=out_split(device), relabel_of=SOURCE)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, path)
    return cid, "done", out["circuit_wall_s"], out["n_candidates_labelled"], out["n_dropped_vs_source"]


def relabel_circuit(src_path, device: str) -> dict:
    """The labelled circuit of `src_path` on `device`: same circuit, candidates and observables."""
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import paper_budget.att_data  # noqa: F401  (puts stage2_predictor on sys.path)
    from quest_phase1.labeling_v2 import (MappedWidthExceeded, device_node_features, make_device,
                                          uncut_reference)
    from quest_phase1.mixed_cuts import label_plan
    from quest_phase1.observables import basket, expectations
    from quest_phase1.simulators import probabilities
    from quest_phase2.data import rebuild_circuit

    src = json.loads(src_path.read_text())
    cid = src["circuit_meta"].get("circuit_id", src_path.stem)
    meta = src["circuit_meta"]
    circ = rebuild_circuit(meta)
    p_ideal = probabilities(circ)
    obs_spec = [tuple(S) for S in src["obs_spec"]]
    if [list(S) for S in basket(circ)] != src["obs_spec"]:
        raise ValueError(f"{cid}: observable basket differs from the stored one")
    obs_ideal = expectations(p_ideal, obs_spec)
    if np.max(np.abs(np.asarray(obs_ideal) - np.asarray(src["obs_ideal"]))) > 1e-9:
        raise ValueError(f"{cid}: rebuilt circuit gives different ideal values")

    ctx = make_device(device, seed=corpus.TRANSPILER_SEED)
    t0 = time.perf_counter()
    ref = uncut_reference(ctx, circ, simulate=False)        # layout only (holdout widths 16-20 > 13, as before)
    records, dropped = [], 0
    for r in src["records"]:
        plan = rebuild_plan(circ, r)
        try:
            rec = label_plan(ctx, circ, plan, p_ideal, None, check_oracle="oracle_tvd" in r,
                             obs_spec=obs_spec, obs_ideal=obs_ideal)
        except MappedWidthExceeded:
            dropped += 1
            continue
        rec["circuit_meta"] = meta
        records.append(rec)
    out = {k: src[k] for k in ("circuit_meta", "n_candidates_total", "n_candidates_by_type", "g_max",
                               "w_max", "max_width", "sample_seed", "label_scheme", "obs_spec",
                               "obs_ideal")}
    out.update({"f_uncut": None, "obs_uncut": None, "uncut_phys": ref["phys"],
                "uncut_n_2q_mapped": ref["n_2q_mapped"], "uncut_depth": ref["depth"],
                "uncut_active_qubits": ref["active_qubits"],
                "n_candidates_labelled": len(records),
                "n_oracle_checked": sum(1 for x in records if "oracle_tvd" in x),
                "n_excluded_mapped_width": src.get("n_excluded_mapped_width", 0) + dropped,
                "n_dropped_vs_source": dropped,
                "n_distinct_overhead": len({round(x["log2_sampling_overhead"], 9) for x in records}),
                "node_features": device_node_features(ctx, circ, ref["phys"]),
                "records": records, "circuit_wall_s": time.perf_counter() - t0})
    return out


def estimate(workers: int) -> None:
    cids = json.loads(SELECTION.read_text())["circuits"]
    walls = sorted((json.loads((DATA_V2 / SOURCE / f"{c}.json").read_text()).get("circuit_wall_s", 0.0)
                    for c in cids), reverse=True)
    tot = sum(walls)
    print(f"fake_brisbane labelling of these {len(walls)} circuits: {tot / 3600:.1f} CPU-hours, "
          f"longest circuit {walls[0] / 60:.0f} min")
    for w in sorted({2, 4, 6, workers}):
        print(f"  ~{max(tot / w, walls[0]) / 3600:.1f} h with {w} workers (the device change may alter this)")


def run(device: str, workers: int) -> None:
    out = DATA_V2 / out_split(device)
    out.mkdir(parents=True, exist_ok=True)
    cids = json.loads(SELECTION.read_text())["circuits"]
    man = DATA_V2 / f"manifest_{out_split(device)}.json"
    if not man.exists():
        m = corpus._manifest("holdout3", str(DATA_V2), dict(corpus.SETTINGS["holdout3"]))
        m.update(split=out_split(device), relabel_of=SOURCE, selection=SELECTION.name,
                 n_circuits=len(cids), label_scheme="v3.1",
                 pythonhashseed=os.environ.get("PYTHONHASHSEED"),
                 procedure="paper_budget.datagen.relabel_device: same circuits and candidates as holdout4")
        from quest_phase1.labeling_v2 import make_device
        ctx = make_device(device, seed=corpus.TRANSPILER_SEED)
        m.update(device=ctx.name, calibration_date=ctx.calibration_date)
        man.write_text(json.dumps(m, indent=1))
    order = sorted(cids, key=lambda c: -json.loads((DATA_V2 / SOURCE / f"{c}.json").read_text())
                   .get("circuit_wall_s", 0.0))                 # longest first
    t0, n_drop = time.time(), 0
    with Pool(workers, maxtasksperchild=2) as pool:
        for i, (cid, st, wall, kept, drop) in enumerate(
                pool.imap_unordered(relabel_one, [(c, device) for c in order]), 1):
            n_drop += drop
            print(f"[{i:3d}/{len(order)}] {st:7s} {cid:<40s} {kept:3d} kept {drop:2d} dropped "
                  f"{wall:7.1f}s  elapsed {(time.time() - t0) / 60:6.1f} min", flush=True)
    done = [c for c in cids if (out / f"{c}.json").exists()]
    print(f"{out_split(device)}: {len(done)}/{len(cids)} circuits, {n_drop} candidates dropped in this run "
          f"(mapped width), {(time.time() - t0) / 3600:.2f} h. Labels NOT summarised: blinded until "
          f"the pre-registered evaluation.")
    print("RELABEL FINISHED" if len(done) == len(cids) else "RELABEL INCOMPLETE: run the same command again")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="fake_sherbrooke", choices=("fake_sherbrooke", "fake_torino"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--estimate", action="store_true", help="runtime estimate only; labels nothing")
    a = ap.parse_args()
    assert os.environ.get("QUEST_UNBLIND") == "holdout4", "set QUEST_UNBLIND=holdout4"
    assert os.environ.get("PYTHONHASHSEED") == "0", "export PYTHONHASHSEED=0 first"
    estimate(a.workers) if a.estimate else run(a.device, a.workers)


if __name__ == "__main__":
    main()
