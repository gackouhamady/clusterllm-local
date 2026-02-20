#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="$ROOT/src/clusterllm/datasets"
RAW_DIR="$ROOT/data/raw"

die(){ echo "ERROR: $*" >&2; exit 1; }
usage(){
  cat >&2 <<EOF
Usage:
  # Prepare + DVC push
  ALLOW_REMOTE_CODE=1 $0 push <dataset|all> <small|large> [seed]

  # DVC pull (restore same files in same locations)
  $0 pull <dataset|all> <small|large>

Examples:
  ALLOW_REMOTE_CODE=1 $0 push mtop_intent small 42
  ALLOW_REMOTE_CODE=1 $0 push all small 42
  $0 pull all small
EOF
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

action="${1:-}"
target="${2:-}"
sc="${3:-}"
seed="${4:-42}"

[[ -z "${action:-}" || -z "${target:-}" || -z "${sc:-}" ]] && { usage; die "Missing args"; }
[[ "$sc" != "small" && "$sc" != "large" ]] && die "scale must be small|large"

PREP_SCRIPT="$ROOT/scripts/run_prepare_data_only.sh"
[[ -x "$PREP_SCRIPT" ]] || die "Missing or not executable: $PREP_SCRIPT (run chmod +x scripts/run_prepare_data_only.sh)"

add_one() {
  local ds="$1"
  local json="$DATASETS_DIR/$ds/$sc.jsonl"
  local train_csv="$RAW_DIR/${ds}_${sc}_train.csv"
  local eval_csv="$RAW_DIR/${ds}_${sc}_eval.csv"

  [[ -f "$json" ]] || die "Missing: $json (did you prepare?)"

  # Track only what exists (safe)
  dvc add "$json"

  [[ -f "$train_csv" ]] && dvc add "$train_csv" || true
  [[ -f "$eval_csv"  ]] && dvc add "$eval_csv"  || true

  # Optional: if your prepare step also leaves these, track them too
  [[ -f "$DATASETS_DIR/$ds/${sc}_train.jsonl" ]] && dvc add "$DATASETS_DIR/$ds/${sc}_train.jsonl" || true
  [[ -f "$DATASETS_DIR/$ds/${sc}_eval.jsonl"  ]] && dvc add "$DATASETS_DIR/$ds/${sc}_eval.jsonl"  || true
}

pull_one() {
  local ds="$1"
  # Pull only what you need (fast)
  dvc pull "$DATASETS_DIR/$ds/$sc.jsonl" || die "dvc pull failed for $ds/$sc"
  [[ -f "$DATASETS_DIR/$ds/$sc.jsonl" ]] || die "After pull, still missing: $DATASETS_DIR/$ds/$sc.jsonl"
}

if [[ "$action" == "push" ]]; then
  mkdir -p "$RAW_DIR" "$DATASETS_DIR"

  if [[ "$target" == "all" ]]; then
    ALLOW_REMOTE_CODE="${ALLOW_REMOTE_CODE:-0}" SEED="$seed" "$PREP_SCRIPT" all "$sc" "$seed"

    # Add every dataset folder that has <scale>.jsonl
    shopt -s nullglob
    for d in "$DATASETS_DIR"/*; do
      [[ -d "$d" ]] || continue
      ds="$(basename "$d")"
      [[ -f "$DATASETS_DIR/$ds/$sc.jsonl" ]] || continue
      add_one "$ds"
    done
    shopt -u nullglob
  else
    ALLOW_REMOTE_CODE="${ALLOW_REMOTE_CODE:-0}" SEED="$seed" "$PREP_SCRIPT" "$target" "$sc" "$seed"
    add_one "$target"
  fi

  # Commit Git metadata (DVC creates *.dvc / updates .gitignore)
  git add -A
  git commit -m "Track prepared data ($target, $sc, seed=$seed)" || true

  # Push data to DVC remote
  dvc push

  echo "DONE: push ($target, $sc)"

elif [[ "$action" == "pull" ]]; then
  if [[ "$target" == "all" ]]; then
    # Pull all tracked outputs (simple)
    dvc pull
    # Safety check: ensure expected files exist where pipeline expects them
    shopt -s nullglob
    for d in "$DATASETS_DIR"/*; do
      [[ -d "$d" ]] || continue
      ds="$(basename "$d")"
      [[ -f "$DATASETS_DIR/$ds/$sc.jsonl" ]] && echo "OK: $DATASETS_DIR/$ds/$sc.jsonl"
    done
    shopt -u nullglob
  else
    pull_one "$target"
  fi

  echo "DONE: pull ($target, $sc)"

else
  usage
  die "action must be push or pull"
fi