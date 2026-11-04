"""Exploratory design analyses on holdout3 (scheme v3.1, already unblinded).

A. Is the device-induced error on <Z_S> mostly multiplicative damping (mu ~ f * mu*)?
B. Does a cut inside the backward light cone of S raise the error on S?
Predictions were committed beforehand in notes/PREDICTIONS_obs_structure.md.
"""
import csv
import json
import os
import sys

import numpy as np

from paper_budget.config import ROOT, DATA_V2, RUNS

sys.path.insert(0, str(ROOT / "stage2_predictor"))
from quest_phase2.data import rebuild_circuit  # noqa: E402  (frozen thesis code)


def light_cone(circ, support):
    """Walk the circuit backwards from the observable's qubits.
    snap[i] = qubits that can still influence S just after gate i.
    in_cone[i] = True if gate i itself can influence S."""
    alive = set(support)
    snap = [None] * len(circ.gates)
    in_cone = np.zeros(len(circ.gates), dtype=bool)
    for i in reversed(range(len(circ.gates))):
        snap[i] = frozenset(alive)
        qubits = set(circ.gates[i].qubits)
        if alive & qubits:
            in_cone[i] = True
            alive |= qubits
    return snap, in_cone


def cuts_in_cone(rec, snap, in_cone) -> int:
    """How many of this candidate's cuts sit inside the light cone."""
    n = sum(bool(in_cone[g]) for g in rec["gate_cut_indices"])
    n += sum(q in snap[t] for q, t in rec["wire_cuts"])    # wire cut just after gate t on qubit q
    return n


def main():
    assert os.environ.get("QUEST_UNBLIND") == "holdout3", "set QUEST_UNBLIND=holdout3"
    out_dir = RUNS / "obs_structure"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []        # (mu_star, mu, n_qubits_in_S) for analysis A
    diffs = []        # per (circuit, observable) light-cone contrast for analysis B
    n_circ = 0
    for path in sorted((DATA_V2 / "holdout3").glob("*.json")):
        if "g1_empty" in path.name:
            continue
        d = json.loads(path.read_text())
        if not d.get("records"):
            continue
        circ = rebuild_circuit(d["circuit_meta"])
        assert len(circ.gates) == len(d["node_features"]), f"rebuild mismatch: {path.name}"
        n_circ += 1
        ideal = np.asarray(d["obs_ideal"], float)
        for s, support in enumerate(d["obs_spec"]):
            snap, in_cone = light_cone(circ, support)
            err_in, err_out = [], []
            for r in d["records"]:
                mu = float(r["obs_mu"][s])
                pairs.append((ideal[s], mu, len(support)))
                err = np.log10((mu - ideal[s]) ** 2 + 1e-12)
                (err_in if cuts_in_cone(r, snap, in_cone) > 0 else err_out).append(err)
            if err_in and err_out:
                diffs.append((path.stem, s, len(support), np.mean(err_in) - np.mean(err_out),
                              len(err_in), len(err_out)))

    P = np.array(pairs)
    mu_star, mu, w = P[:, 0], P[:, 1], P[:, 2]

    print(f"circuits: {n_circ}   (candidate, observable) pairs: {len(P)}")
    print("\n=== A. multiplicative damping ===")
    big = np.abs(mu_star) > 0.2
    ratio = mu[big] / mu_star[big]
    print(f"|mu*|>0.2: n={big.sum()}  ratio quantiles 5/25/50/75/95% = "
          f"{np.round(np.quantile(ratio, [.05, .25, .5, .75, .95]), 3)}")
    print(f"A1 share of ratios in [0, 1.05]: {np.mean((ratio >= 0) & (ratio <= 1.05)):.3f}")
    small1 = (np.abs(mu_star) < 0.02) & (w == 1)
    print(f"A2 single-Z with |mu*|<0.02: n={small1.sum()}  median mu = {np.median(mu[small1]):+.4f}")
    sq = (mu - mu_star) ** 2
    for lo, hi in [(0, .02), (.02, .2), (.2, .6), (.6, 1.01)]:
        m = (np.abs(mu_star) >= lo) & (np.abs(mu_star) < hi)
        print(f"share of total squared error from |mu*| in [{lo},{hi}): {sq[m].sum() / sq.sum():.3f}")

    print("\n=== B. light cone ===")
    D = np.array([x[3] for x in diffs])
    print(f"(circuit, observable) pairs with both kinds of candidates: {len(D)}")
    print(f"median log10 error difference (in - out): {np.median(D):+.3f}")
    print(f"B1 share > 0: {np.mean(D > 0):.3f}")
    try:
        from scipy.stats import wilcoxon
        print(f"Wilcoxon one-sided (in > out) p = {wilcoxon(D, alternative='greater').pvalue:.2e}")
    except ImportError:
        print("scipy not available: Wilcoxon skipped")

    with open(out_dir / "light_cone_pairs.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["circuit", "obs_index", "obs_weight", "dlog10err_in_minus_out", "n_in", "n_out"])
        wr.writerows(diffs)
    print(f"\nper-pair table written to {out_dir / 'light_cone_pairs.csv'}")


if __name__ == "__main__":
    main()
