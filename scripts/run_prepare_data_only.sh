#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

PREP="$ROOT/src/data/prepare_data.py"
DATA_DIR="$ROOT/src/clusterllm/datasets"
RAW_DIR="$ROOT/data/raw"

SEED="${SEED:-42}"

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat >&2 <<'EOF'
Usage:
  # One dataset:
  ALLOW_REMOTE_CODE=1 scripts/run_prepare_data_only.sh <dataset> <small|large> [seed]

  # All datasets (prepare all tasks defined in prepare_data.py):
  ALLOW_REMOTE_CODE=1 scripts/run_prepare_data_only.sh all <small|large> [seed]

Examples:
  ALLOW_REMOTE_CODE=1 scripts/run_prepare_data_only.sh mtop_intent small
  ALLOW_REMOTE_CODE=1 scripts/run_prepare_data_only.sh bank77 large 42
  ALLOW_REMOTE_CODE=1 scripts/run_prepare_data_only.sh all small 42

Notes:
  - MTOP requires: ALLOW_REMOTE_CODE=1
  - Outputs created:
      data/raw/<name>_<scale>_train.csv
      data/raw/<name>_<scale>_eval.csv
      src/clusterllm/datasets/<name>/<scale>.jsonl   (created by concatenating *_train.jsonl + *_eval.jsonl)
EOF
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

target="${1:-}"
sc="${2:-}"
seed_arg="${3:-}"

[[ -z "${target:-}" || -z "${sc:-}" ]] && { usage; die "Missing args: <dataset|all> <small|large> [seed]"; }

if [[ "$sc" != "small" && "$sc" != "large" ]]; then
  die "scale must be small|large, got: $sc"
fi

if [[ -n "${seed_arg:-}" ]]; then
  SEED="$seed_arg"
fi

mkdir -p "$RAW_DIR" "$DATA_DIR"

prepare_one() {
  local ds="$1"
  mkdir -p "$DATA_DIR/$ds"

  local out_train_jsonl="$DATA_DIR/$ds/${sc}_train.jsonl"
  local out_eval_jsonl="$DATA_DIR/$ds/${sc}_eval.jsonl"
  local out_final_jsonl="$DATA_DIR/$ds/${sc}.jsonl"

  echo "==> PREPARE: ds=$ds sc=$sc seed=$SEED"

  python "$PREP" \
    --dataset "$ds" \
    --seed "$SEED" \
    --output-dir "$RAW_DIR" \
    --datasets-dir "$DATA_DIR" \
    --split-train "$sc" \
    --split-eval "$sc" \
    --out-train "$RAW_DIR/${ds}_${sc}_train.csv" \
    --out-eval "$RAW_DIR/${ds}_${sc}_eval.csv" \
    --jsonl-train "$out_train_jsonl" \
    --jsonl-eval "$out_eval_jsonl"

  # Build the file expected by the rest of the pipeline: <scale>.jsonl
  cat "$out_train_jsonl" "$out_eval_jsonl" > "$out_final_jsonl"
  test -f "$out_final_jsonl" || die "Still missing $out_final_jsonl"

  echo "OK: $out_final_jsonl"
}

if [[ "$target" == "all" ]]; then
  echo "==> PREPARE ALL DATASETS (as defined inside prepare_data.py) sc=$sc seed=$SEED"

  # Run prepare_data once in "all tasks" mode.
  # This will create:
  #   src/clusterllm/datasets/<ds>/large.jsonl and small.jsonl (based on its internal defaults)
  # But your pipeline expects <scale>.jsonl (small.jsonl or large.jsonl), so we harmonize afterward.
  #
  # Important: we do NOT pass --dataset here so it runs all tasks.
  python "$PREP" \
    --seed "$SEED" \
    --output-dir "$RAW_DIR" \
    --datasets-dir "$DATA_DIR" \
    --split-train "$sc" \
    --split-eval "$sc"

  # Harmonize: for each dataset folder created, ensure <scale>.jsonl exists.
  # Two cases:
  #  (1) If prepare_data already wrote $DATA_DIR/<ds>/$sc.jsonl, keep it.
  #  (2) Else if it wrote ${sc}_train.jsonl + ${sc}_eval.jsonl, concatenate.
  #  (3) Else if it wrote "large.jsonl/small.jsonl", copy/rename to "$sc.jsonl".
  shopt -s nullglob
  for d in "$DATA_DIR"/*; do
    [[ -d "$d" ]] || continue
    ds="$(basename "$d")"

    # skip non-dataset dirs if any
    [[ "$ds" == "__pycache__" ]] && continue

    final="$d/$sc.jsonl"
    [[ -f "$final" ]] && { echo "OK (exists): $final"; continue; }

    train="$d/${sc}_train.jsonl"
    eval="$d/${sc}_eval.jsonl"
    if [[ -f "$train" && -f "$eval" ]]; then
      cat "$train" "$eval" > "$final"
      echo "OK (cat train+eval): $final"
      continue
    fi

    # fallback: internal default outputs large.jsonl/small.jsonl
    if [[ -f "$d/large.jsonl" && "$sc" == "large" ]]; then
      cp -f "$d/large.jsonl" "$final"
      echo "OK (cp large.jsonl): $final"
      continue
    fi
    if [[ -f "$d/small.jsonl" && "$sc" == "small" ]]; then
      cp -f "$d/small.jsonl" "$final"
      echo "OK (cp small.jsonl): $final"
      continue
    fi

    echo "WARN: could not build $final (missing expected files in $d)" >&2
  done
  shopt -u nullglob

  echo "DONE: prepare all (requested scale=$sc)."

else
  prepare_one "$target"
fi