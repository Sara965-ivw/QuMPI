"""Data for the residual attenuation model (notes/PLAN_gt_att.md, sections 1.1, 1.2 and 3).

Wraps the frozen thesis loader (quest_phase2.data.build_graph, aggregation "layout") and attaches,
per circuit, the quantities the attenuation models and the budget-aware evaluation need:
    att  (K,)  a_k = mean over informative observables of -ln(mu_kS / mu*_S)   (headroom.load)
    e    (K,)  -ln ESP_k, layout-ESP of quest_phase2.hardware                   (model prior input)
    x    (K,7) cut features, as in headroom.FEATS                               (linear correction)
    g2, b2, v (K,), m (scalar)                                                  (evaluation only)
No file in stage2_predictor/ is modified. holdout3 stays behind the frozen QUEST_UNBLIND guard.

    python -m paper_budget.att_data --summary train_v31      # what is usable for training
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paper_budget.config import ROOT, DATA_V2, SPLIT_SEED
from paper_budget.targets import bias_variance, gamma2

sys.path.insert(0, str(ROOT / "stage2_predictor"))
from quest_phase2.data import build_graph, stratified_split, _blind_guard  # noqa: E402  (frozen)

BIG = 0.2                     # informative observable: |mu*| > BIG (same as headroom.py)
RATIO_FLOOR = 1e-6            # same clip as headroom.load, keeps -ln finite
FEATS = ("g", "w", "log2_sampling_overhead", "routing_overhead_2q",
         "n_2q_mapped", "depth_a", "depth_b")


def attenuation(mu: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    """(K, S) noisy and (S,) ideal expectation values -> (K,) attenuation labels.

    a_k = mean_{S in I} -ln(mu_kS / mu*_S), I = {S : |mu*_S| > BIG}; NaN if I is empty.
    If mu_kS = f_k mu*_S exactly, a_k = -ln f_k.
    """
    mu, ideal = np.asarray(mu, float), np.asarray(ideal, float)
    inf = np.abs(ideal) > BIG
    if not inf.any():
        return np.full(len(mu), np.nan)
    return (-np.log(np.clip(mu[:, inf] / ideal[inf], RATIO_FLOOR, None))).mean(axis=1)


@dataclass
class AttCircuit:
    g: object                 # quest_phase2.data.CircuitGraph (frozen features for the Transformer)
    att: np.ndarray
    e: np.ndarray
    x: np.ndarray
    g2: np.ndarray
    b2: np.ndarray
    v: np.ndarray
    m: float
    n_informative: int
    esp_missing: int

    @property
    def cid(self) -> str:
        return self.g.cid

    @property
    def n_cand(self) -> int:
        return len(self.att)

    def usable_train(self) -> bool:
        """>= 2 candidates, a defined label, a layout-ESP for every candidate, and some spread."""
        return (self.n_cand >= 2 and np.all(np.isfinite(self.att)) and self.esp_missing == 0
                and float(np.ptp(self.att)) > 0)

    def usable_eval(self) -> bool:
        """Same filter as deployable_rule.py and residual_att.py (>= 3 candidates)."""
        return self.n_cand >= 3 and np.all(np.isfinite(self.att)) and self.esp_missing == 0


def load_circuit(path: Path, split: str) -> AttCircuit | None:
    d = json.loads(Path(path).read_text())
    if not d.get("records") or "obs_ideal" not in d:
        return None
    g = build_graph(path, split, aggregation="layout")
    if g is None:
        return None
    recs = d["records"]
    ideal = np.asarray(d["obs_ideal"], float)
    mu = np.array([r["obs_mu"] for r in recs], float)
    bv = bias_variance(d)
    log_esp = np.asarray(g.esp, float)
    return AttCircuit(g=g, att=attenuation(mu, ideal), e=-log_esp,
                      x=np.array([[float(r[f]) for f in FEATS] for r in recs]),
                      g2=np.array([gamma2(r) for r in recs], float),
                      b2=bv[:, 0], v=bv[:, 1], m=float(np.mean(ideal ** 2)),
                      n_informative=int(np.sum(np.abs(ideal) > BIG)),
                      esp_missing=int(np.sum(log_esp == 0.0)))


def load_att_split(name: str, split: str | None = None) -> list[AttCircuit]:
    corpus = DATA_V2 / name
    _blind_guard(corpus)                                  # frozen holdout3 guard
    out = []
    for p in sorted(corpus.glob("*.json")):
        if p.name.endswith(".g1_empty.json"):
            continue
        c = load_circuit(p, split or name)
        if c is not None:
            out.append(c)
    return out


def split_train_val(circuits: list[AttCircuit], seed: int = SPLIT_SEED) -> tuple[list, list]:
    """Circuit-level split stratified by (family, width), frozen quest_phase2.stratified_split."""
    by_id = {c.cid: c for c in circuits}
    assert len(by_id) == len(circuits), "duplicate circuit ids"
    tr, va = stratified_split([c.g for c in circuits], val_frac=0.2, seed=seed)
    train, val = [by_id[g.cid] for g in tr], [by_id[g.cid] for g in va]
    assert not ({c.cid for c in train} & {c.cid for c in val}), "train/val overlap"
    return train, val


def summary(name: str) -> None:
    files = [p for p in (DATA_V2 / name).glob("*.json") if not p.name.endswith(".g1_empty.json")]
    C = load_att_split(name)
    tr = [c for c in C if c.usable_train()]
    ev = [c for c in C if c.usable_eval()]
    att = np.concatenate([c.att for c in tr]) if tr else np.array([np.nan])
    print(f"{name}: {len(files)} files, {len(C)} with candidates and observables")
    print(f"  usable for training (>= 2 cands, label, ESP, spread): {len(tr)} circuits, "
          f"{sum(c.n_cand for c in tr)} candidates")
    print(f"  usable for evaluation (>= 3 cands, label, ESP):       {len(ev)} circuits")
    print(f"  without informative observable: {sum(c.n_informative == 0 for c in C)}; "
          f"with missing layout-ESP: {sum(c.esp_missing > 0 for c in C)}")
    print(f"  attenuation label: min {np.nanmin(att):+.4f}, median {np.nanmedian(att):+.4f}, "
          f"max {np.nanmax(att):+.4f}, share < 0: {np.mean(att < 0):.3f}")
    fam = Counter((c.g.family, c.g.n_qubits) for c in tr)
    print("  training circuits by (family, qubits): "
          + ", ".join(f"{f}/{n}: {k}" for (f, n), k in sorted(fam.items())))
    if len(tr) >= 2:
        a, b = split_train_val(tr)
        print(f"  stratified split (seed {SPLIT_SEED}): train {len(a)}, validation {len(b)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="corpus folder, e.g. train_v31")
    summary(ap.parse_args().summary)
