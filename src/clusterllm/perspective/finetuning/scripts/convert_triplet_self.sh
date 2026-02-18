#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/hamadygackou777/clusterllm-local"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
cd "${REPO_ROOT}/src/clusterllm/perspective/finetuning"

# Defaults (override via args)
DATASET="banking77"
RUN_DIR="${REPO_ROOT}/runs/bank77_small/deepseek_finetuning"
TRIPLETS_FILE="${RUN_DIR}/triplets/triplets.json"
DATA_PATH="${REPO_ROOT}/datasets/banking77/test.jsonl"
FEAT_PATH="${REPO_ROOT}/datasets/banking77/embeddings.hdf5"
OUT_DIR="${RUN_DIR}/finetuning_data"

usage() {
  echo "Usage: $0 [--dataset DS] [--run_dir DIR] [--triplets_file FILE] [--data_path FILE] --feat_path FILE [--out_dir DIR]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --run_dir) RUN_DIR="$2"; shift 2;;
    --triplets_file) TRIPLETS_FILE="$2"; shift 2;;
    --data_path) DATA_PATH="$2"; shift 2;;
    --feat_path) FEAT_PATH="$2"; shift 2;;
    --out_dir) OUT_DIR="$2"; shift 2;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

mkdir -p "$OUT_DIR"

if [[ ! -f "$TRIPLETS_FILE" ]]; then
  echo "❌ ERREUR: triplets introuvable: $TRIPLETS_FILE"
  exit 1
fi

if [[ ! -f "$FEAT_PATH" ]]; then
  echo "❌ ERREUR: embeddings introuvable: $FEAT_PATH"
  echo "   Il faut un .h5/.hdf5 contenant la clé 'embeds'."
  exit 1
fi

echo "🚀 Conversion Self (embeddings-based)"
echo "  dataset       : $DATASET"
echo "  triplets_file : $TRIPLETS_FILE"
echo "  data_path     : $DATA_PATH"
echo "  feat_path     : $FEAT_PATH"
echo "  out_dir       : $OUT_DIR"

python convert_triplet_self.py \
  --dataset "$DATASET" \
  --pred_path "$TRIPLETS_FILE" \
  --data_path "$DATA_PATH" \
  --feat_path "$FEAT_PATH" \
  --output_path "$OUT_DIR"
