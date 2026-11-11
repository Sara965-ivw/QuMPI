"""Write the holdout4 analysis set as a circuit list for the baseline tools.

The list is the pre-registered analysis set (notes/PREREG_holdout4.md section 3,
notes/holdout4_n156_resolution.md): every labelled holdout4 circuit with >= 3 candidates,
>= 1 informative observable and a layout-ESP for every candidate, i.e. exactly the circuits
eval_holdout4.py evaluated (eval_gt_att.load_test). find_cuts and Qdislib are run on this list
so that every method is compared on the same circuits as the confirmatory test.

  QUEST_UNBLIND=holdout4 PYTHONHASHSEED=0 python -m paper_budget.holdout4_list

Output: data/corpus_v2/holdout4_analysis_set.json. An existing file is never overwritten;
a rerun only checks that it lists the same circuits.
"""
from __future__ import annotations

import hashlib
import json
import os

from paper_budget.config import DATA_V2, RUNS
from paper_budget.eval_gt_att import load_test

OUT = DATA_V2 / "holdout4_analysis_set.json"
EVAL = RUNS / "gt_att" / "eval4_holdout4.json"


def main() -> None:
    assert os.environ.get("QUEST_UNBLIND") == "holdout4", "set QUEST_UNBLIND=holdout4"
    assert os.environ.get("PYTHONHASHSEED") == "0", "export PYTHONHASHSEED=0 first"
    C, _ = load_test("holdout4")
    cids = sorted(c.cid for c in C)
    n_eval = json.loads(EVAL.read_text())["n_circuits"]
    assert len(cids) == n_eval, f"{len(cids)} circuits now vs {n_eval} in {EVAL.name}"
    digest = hashlib.sha256("\n".join(cids).encode()).hexdigest()
    if OUT.exists():
        old = json.loads(OUT.read_text())
        assert old["circuits"] == cids, f"{OUT} exists with a different circuit list"
        print(f"{OUT.name} already exists and matches ({len(cids)} circuits, sha256 {digest[:12]})")
        return
    OUT.write_text(json.dumps({
        "rule": "PREREG_holdout4.md section 3: >= 3 candidates, >= 1 informative observable, "
                "layout-ESP for every candidate (eval_gt_att.load_test)",
        "n_circuits": len(cids), "sha256_of_ids": digest, "circuits": cids}, indent=1))
    print(f"written {OUT} ({len(cids)} circuits, sha256 {digest[:12]})")


if __name__ == "__main__":
    main()
