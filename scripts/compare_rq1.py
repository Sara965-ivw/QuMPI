#!/usr/bin/env python3
"""Compare the frozen primary/GBM gap-weighted ranking values after a reproduction."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

MODELS = {"gt_size-layout", "gbm-layout"}
KEYS = ("model", "cid", "seed")
METRIC = "kendall_w"

def load(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    selected = {}
    for row in rows:
        if row.get("model") not in MODELS:
            continue
        key = tuple(row[k] for k in KEYS)
        if key in selected:
            raise ValueError(f"duplicate key {key} in {path}")
        selected[key] = row.get(METRIC, "")
    return selected

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--reproduced',type=Path,required=True)
    a=ap.parse_args()
    try:
        base=load(a.baseline); new=load(a.reproduced)
    except Exception as exc:
        print(f"ERROR: could not read comparison CSV: {exc}",file=sys.stderr); return 2
    issues=[]
    if set(base)!=set(new):
        missing=sorted(set(base)-set(new))[:5]; extra=sorted(set(new)-set(base))[:5]
        issues.append(f"row-key mismatch: missing={missing}; extra={extra}")
    for key in sorted(set(base)&set(new)):
        if base[key] != new[key]:
            issues.append(f"value mismatch {key}: baseline={base[key]} reproduced={new[key]}")
            if len(issues)>=20: break
    if issues:
        print(f"RQ1 reproduction comparison FAILED ({len(issues)} issue(s), selected rows {len(new)}):")
        for issue in issues: print(' -',issue)
        return 1
    expected=143*5*len(MODELS)
    if len(base)!=expected:
        print(f"WARNING: compared {len(base)} rows, expected {expected}; review the frozen artifact set.")
        return 1
    print(f"RQ1 reproduction matches exactly: {len(base)} gap-weighted Kendall values across {sorted(MODELS)}.")
    return 0

if __name__=='__main__':
    raise SystemExit(main())
