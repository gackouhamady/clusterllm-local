#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash scripts/sweep_full_pipeline_2llms.sh queue configs/grid_100.txt
#   bash scripts/sweep_full_pipeline_2llms.sh run   configs/grid_100.txt 4
#   bash scripts/sweep_full_pipeline_2llms.sh repro
#   bash scripts/sweep_full_pipeline_2llms.sh show
#   bash scripts/sweep_full_pipeline_2llms.sh clean

STAGE="full_pipeline_2llms"
MODE="${1:-}"
GRID_FILE="${2:-configs/grid_100.txt}"
JOBS="${3:-4}"

die(){ echo "ERROR: $*" >&2; exit 1; }

if [[ -z "${MODE}" || "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  cat >&2 <<EOF
Usage:
  $0 queue <grid_file>
  $0 run   <grid_file> [jobs]
  $0 repro
  $0 show
  $0 clean

Notes:
- 'queue' : add all scenarios to DVC queue (no execution)
- 'run'   : queue + run all queued experiments with parallel jobs
- 'repro' : run classic dvc repro for the stage (no grid)
- 'show'  : show experiments table
- 'clean' : remove queued experiments
EOF
  exit 0
fi

case "$MODE" in
  queue|run)
    [[ -f "$GRID_FILE" ]] || die "grid file not found: $GRID_FILE"

    echo "==> Queueing scenarios from: $GRID_FILE"
    # Read grid, ignore blank lines + comments
    while IFS= read -r line; do
      [[ -z "${line// /}" ]] && continue
      [[ "${line}" =~ ^[[:space:]]*# ]] && continue

      # Parse 5 fields
      read -r ds sc lt lp seed <<<"$line"
      [[ -n "${ds:-}" && -n "${sc:-}" && -n "${lt:-}" && -n "${lp:-}" && -n "${seed:-}" ]] \
        || die "bad line (need 5 fields): $line"

      dvc exp run --queue -s "$STAGE" \
        -S run.dataset="$ds" \
        -S run.scale="$sc" \
        -S run.llm_triplet="$lt" \
        -S run.llm_pairs="$lp" \
        -S run.seed="$seed"
    done < "$GRID_FILE"

    echo "==> Queue done."
    if [[ "$MODE" == "run" ]]; then
      echo "==> Running ALL queued experiments with jobs=$JOBS"
      dvc exp run --run-all --jobs "$JOBS"
    else
      echo "==> To execute: dvc exp run --run-all --jobs $JOBS"
    fi
    ;;

  repro)
    echo "==> dvc repro -s $STAGE"
    dvc repro -s "$STAGE"
    ;;

  show)
    echo "==> dvc exp show"
    dvc exp show
    ;;

  clean)
    echo "==> Removing queued experiments"
    dvc exp remove --queue -f
    ;;

  *)
    die "unknown mode: $MODE"
    ;;
esac