"""Degradation by width on holdout4 (EXPLORATORY, descriptive).

The proposal's RQ1 design asks how prediction quality changes with circuit width. This script
answers it for the tested widths (16, 18, 20 qubits) from the output of compare_holdout4.py only:
no model, label, find_cuts or Qdislib run is repeated, and nothing is refitted.

Input : runs/paper_budget/compare_holdout4/per_circuit.json (151 circuits; the 5 circuits on which
        find_cuts failed are absent, as in the tool comparison)
Output: runs/paper_budget/compare_holdout4/by_width.json, and the tables printed below

  python -m paper_budget.analysis.width_holdout4

Per width:
  - regret of M2b, M1, M0 and the minimum-overhead rule against the best CANDIDATE (as eval_holdout4.py)
  - error of every method against the best available cut (candidates + tools' cuts, as compare_holdout4.py)
  - at 1e9 shots: ratios M2b/M1, M2b/M0 (< 1 = M2b better) and find_cuts/M2b, Qdislib/M2b (> 1 = M2b
    better), geometric mean over circuits with a 95% bootstrap CI (10 000 resamples, seed 2026)
No p-values are computed: with about 50 circuits per width the comparison is descriptive.
"""
from __future__ import annotations

import json

import numpy as np

from paper_budget.config import BUDGETS, RUNS

SRC = RUNS / "compare_holdout4" / "per_circuit.json"
OUT = RUNS / "compare_holdout4" / "by_width.json"
METHODS = ("M2b", "M1", "M0", "min_kappa2", "metis", "random", "find_cuts", "qdislib")
OURS = ("M2b", "M1", "M0", "min_kappa2")
PAIRS = (("M2b", "M1"), ("M2b", "M0"), ("find_cuts", "M2b"), ("qdislib", "M2b"))
J9 = len(BUDGETS) - 1                     # index of 1e9
SHOW = (0, 3, J9)                         # 1e3, 1e6, 1e9
N_BOOT, SEED = 10_000, 2026


def gm_ratio(rows: list[dict], num: str, den: str) -> np.ndarray:
    """Geometric mean over circuits of num / den, per budget (7,)."""
    L = [np.log(np.asarray(r[num], float) / np.asarray(r[den], float)) for r in rows]
    return np.exp(np.mean(L, axis=0))


def ratio_ci(rows: list[dict], num: str, den: str, j: int, rng) -> dict:
    """Geometric-mean ratio num / den at budget index j, with a circuit bootstrap CI."""
    L = np.array([np.log(r[num][j] / r[den][j]) for r in rows])
    idx = rng.integers(0, len(L), size=(N_BOOT, len(L)))
    lo, hi = np.quantile(np.exp(L[idx].mean(axis=1)), [0.025, 0.975])
    return {"n": int(len(L)), "ratio": float(np.exp(L.mean())), "ci95": [float(lo), float(hi)]}


def main() -> None:
    rows = json.loads(SRC.read_text())
    widths = sorted({int(r["n_qubits"]) for r in rows})
    rng = np.random.default_rng(SEED)
    out = {"source": str(SRC.relative_to(RUNS.parent.parent)), "n_circuits": len(rows), "widths": {}}
    for w in widths:
        rs = [r for r in rows if int(r["n_qubits"]) == w]
        res = {"n": len(rs), "n_connected": sum(not r["separable"] for r in rs),
               "regret_vs_best_candidate": {m: gm_ratio(rs, m, "oracle_cand").tolist() for m in OURS},
               "vs_best_available_cut": {}, "ratios_1e9": {}}
        for m in METHODS:
            have = [r for r in rs if m in r]
            res["vs_best_available_cut"][m] = {"n": len(have), "gm": gm_ratio(have, m, "oracle").tolist()}
        for num, den in PAIRS:
            have = [r for r in rs if num in r and den in r]
            res["ratios_1e9"][f"{num}/{den}"] = ratio_ci(have, num, den, J9, rng)
        out["widths"][str(w)] = res

    print(f"holdout4 by width ({len(rows)} circuits; source {SRC.name})\n")
    print("Regret against the best candidate at 1e9 shots")
    print(f"{'width':>6}{'n':>5}" + "".join(f"{m:>12}" for m in OURS))
    for w, res in out["widths"].items():
        print(f"{w:>6}{res['n']:>5}" + "".join(f"{res['regret_vs_best_candidate'][m][J9]:>12.3f}" for m in OURS))

    print("\nError against the best available cut (candidates and tools' cuts), at 1e3 / 1e6 / 1e9 shots")
    for w, res in out["widths"].items():
        print(f"-- {w} qubits (n = {res['n']}, connected {res['n_connected']})")
        for m in METHODS:
            v = res["vs_best_available_cut"][m]
            print(f"   {m:<12}" + "".join(f"{v['gm'][j]:>9.3f}" for j in SHOW))

    print("\nRatios at 1e9 shots, geometric mean (95% bootstrap CI)")
    print(f"{'width':>6}" + "".join(f"{a + '/' + b:>26}" for a, b in PAIRS))
    for w, res in out["widths"].items():
        cells = []
        for a, b in PAIRS:
            t = res["ratios_1e9"][f"{a}/{b}"]
            cells.append(f"{t['ratio']:.3f} [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]")
        print(f"{w:>6}" + "".join(f"{c:>26}" for c in cells))

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
