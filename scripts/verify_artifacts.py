#!/usr/bin/env python3
"""Read-only SHA-256 verification of frozen QuMPI data and result artifacts."""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data_manifest" / "FROZEN_ARTIFACTS_SHA256SUMS.txt"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    if not MANIFEST.is_file():
        print(f"ERROR: checksum manifest missing: {MANIFEST}", file=sys.stderr)
        return 2
    failures = []
    checked = 0
    for lineno, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            expected, rel = line.split(None, 1)
            rel = rel.strip().lstrip("*")
        except ValueError:
            failures.append(f"line {lineno}: malformed manifest row")
            continue
        path = (ROOT / rel).resolve()
        if ROOT.resolve() not in path.parents:
            failures.append(f"line {lineno}: path escapes repository: {rel}")
            continue
        if not path.is_file():
            failures.append(f"MISSING: {rel}")
            continue
        actual = sha256(path)
        checked += 1
        if actual != expected:
            failures.append(f"HASH MISMATCH: {rel} expected={expected} actual={actual}")
    if failures:
        print(f"FAILED: checked {checked} files; {len(failures)} issue(s)")
        for item in failures:
            print(item)
        return 1
    print(f"OK: {checked} frozen artifact files match SHA-256 manifest.")
    print("This check does not load or analyse holdout data and does not run experiments.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
