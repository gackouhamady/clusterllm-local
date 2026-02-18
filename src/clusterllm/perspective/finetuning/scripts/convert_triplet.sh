#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/hamadygackou777/clusterllm-local"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
cd "${REPO_ROOT}/src/clusterllm/perspective/finetuning"

# Defaults (tu peux override via args)
DATASET="banking77"
RUN_DIR="${REPO_ROOT}/runs/bank77_small/deepseek_finetuning"
MODEL_NAME="deepseek-r1_32b"
DATA_PATH="${REPO_ROOT}/datasets/banking77/test.jsonl"
OUT_DIR="${RUN_DIR}/finetuning_data"
PRED_FILE=""

usage() {
  echo "Usage: $0 [--dataset DS] [--run_dir DIR] [--model_name NAME] [--pred_file FILE] [--data_path FILE] [--out_dir DIR] [--e5]"
  exit 1
}

E5_FLAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --run_dir) RUN_DIR="$2"; shift 2;;
    --model_name) MODEL_NAME="$2"; shift 2;;
    --pred_file) PRED_FILE="$2"; shift 2;;
    --data_path) DATA_PATH="$2"; shift 2;;
    --out_dir) OUT_DIR="$2"; shift 2;;
    --e5) E5_FLAG="--e5"; shift 1;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

mkdir -p "$OUT_DIR"

# Default pred file pattern (adapté à ton repo: runs/.../triplets/triplets-<model>-pred.json)
if [[ -z "${PRED_FILE}" ]]; then
  PRED_FILE="${RUN_DIR}/triplets/triplets-${MODEL_NAME}-pred.json"
fi

if [[ ! -f "$PRED_FILE" ]]; then
  echo "❌ ERREUR: fichier de prédiction introuvable: $PRED_FILE"
  echo "   Indique-le explicitement avec --pred_file, ou vérifie --run_dir/--model_name."
  exit 1
fi

echo "🚀 Conversion Standard"
echo "  dataset   : $DATASET"
echo "  pred_file : $PRED_FILE"
echo "  data_path : $DATA_PATH"
echo "  out_dir   : $OUT_DIR"

python convert_triplet.py \
  --dataset "$DATASET" \
  --pred_path "$PRED_FILE" \
  --data_path "$DATA_PATH" \
  --output_path "$OUT_DIR" \
  $E5_FLAG
