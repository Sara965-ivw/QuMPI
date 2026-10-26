#!/usr/bin/env bash
# QuMPI safe task runner. No auto-install, deletion, commit, push, or tag.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
cmd="${1:-help}"
usage() {
  cat <<'EOF'
Usage:
  scripts/quest.sh verify-artifacts
  scripts/quest.sh analyze-rq2
  scripts/quest.sh reproduce-rq1 --confirm

verify-artifacts   Read-only SHA-256 verification; no experiments.
analyze-rq2        Re-analyse the existing RQ2 JSONL in a new output folder; no MPI simulation.
reproduce-rq1      Explicitly opt-in to re-evaluating the already-unblinded holdout3;
                   writes only under a new runs/reproduce_rq1_<UTC timestamp>/ directory.

This runner never installs dependencies, commits, pushes, tags, or deletes existing outputs.
EOF
}
case "$cmd" in
  help|-h|--help)
    usage
    ;;
  verify-artifacts)
    exec python3 "$ROOT/scripts/verify_artifacts.py"
    ;;
  analyze-rq2)
    RUNS="$ROOT/runs/rq2_prelim/runs.jsonl"
    [[ -f "$RUNS" ]] || { echo "ERROR: missing $RUNS" >&2; exit 2; }
    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    OUT="$ROOT/runs/rq2_reanalysis_$STAMP"
    [[ ! -e "$OUT" ]] || { echo "ERROR: output already exists: $OUT" >&2; exit 2; }
    mkdir -p "$OUT"
    echo "Analysing captured pilot data only; no MPI jobs will be launched. Output: $OUT"
    python3 "$ROOT/stage4_mpi/rq2_analyse.py" --runs "$RUNS" --out "$OUT"
    cmp -s "$OUT/per_run.csv" "$ROOT/runs/rq2_prelim/per_run.csv" || { echo "ERROR: RQ2 per_run.csv differs from the archived analysis; retain both outputs for review." >&2; exit 1; }
    cmp -s "$OUT/summary.txt" "$ROOT/runs/rq2_prelim/summary.txt" || { echo "ERROR: RQ2 summary.txt differs from the archived analysis; retain both outputs for review." >&2; exit 1; }
    echo "RQ2 post-processing matches the archived per_run.csv and summary.txt exactly. PNG byte equality is not required."
    ;;
  reproduce-rq1)
    [[ "$#" -eq 2 && "${2:-}" == "--confirm" ]] || { echo "ERROR: RQ1 reproduction re-reads the already-unblinded holdout3. Pass only --confirm explicitly." >&2; usage; exit 2; }
    python3 "$ROOT/scripts/verify_artifacts.py"
    FROZEN="$ROOT/runs/rq1_h3"
    CORPUS="$ROOT/data/corpus_v2"
    [[ -d "$CORPUS/train" && -d "$CORPUS/holdout3" && -f "$CORPUS/holdout3_selection.json" ]] || { echo "ERROR: required corpus paths missing" >&2; exit 2; }
    [[ -f "$FROZEN/partB_record.json" ]] || { echo "ERROR: frozen weight record missing" >&2; exit 2; }
    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    OUT="$ROOT/runs/reproduce_rq1_$STAMP"
    [[ ! -e "$OUT" ]] || { echo "ERROR: output already exists: $OUT" >&2; exit 2; }
    mkdir -p "$OUT"
    cp "$FROZEN/partB_record.json" "$OUT/partB_record.json"
    for f in "$FROZEN"/gt_size-layout_seed*.pt "$FROZEN"/gt_size-layout_uniform_seed*.pt; do
      [[ -f "$f" ]] || { echo "ERROR: missing expected checkpoint: $f" >&2; exit 2; }
      cp "$f" "$OUT/"
    done
    echo "Evaluating previously unblinded holdout3. Existing runs/rq1_h3 will not be modified."
    echo "New output: $OUT"
    QUEST_UNBLIND=holdout3 python3 "$ROOT/stage2_predictor/run_rq1_h3.py" \
      --stage evaluate --train "$CORPUS" --holdout3 "$CORPUS/holdout3" --out "$OUT"
    python3 "$ROOT/scripts/compare_rq1.py" --baseline "$FROZEN/per_circuit.csv" --reproduced "$OUT/per_circuit.csv"
    ;;
  *)
    echo "Unknown task: $cmd" >&2
    usage
    exit 2
    ;;
esac
