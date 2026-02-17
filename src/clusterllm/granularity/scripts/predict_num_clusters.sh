#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/hamadygackou777/clusterllm-local"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
cd "${REPO_ROOT}/src/clusterllm/granularity"

MODEL_NAME="${1:-"qwen2.5:7b"}"
DATASET="${2:-"banking77"}"

echo "================================================================"
echo "📊 DÉTERMINATION DE LA GRANULARITÉ (STAGE 2 - MODE K=3)"
echo "================================================================"

# FORCE LE CHEMIN VERS LE FICHIER K=3
INPUT_FILE="sampled_pair_results/${DATASET}_${MODEL_NAME}/${DATASET}_embed=finetuned_s=small_k=3_multigran2-200_seed=100.json"
DATA_PATH="${REPO_ROOT}/datasets/${DATASET}/test.jsonl"
OUT_DIR="predicted_num_clusters_results/${DATASET}_${MODEL_NAME}"
mkdir -p "$OUT_DIR"
OUTPUT_FILE="${OUT_DIR}/final_granularity_results.json"

if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ ERREUR : Le fichier K=3 est introuvable à : $INPUT_FILE"
    exit 1
fi

python3 predict_num_clusters.py \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --clustering_results "$INPUT_FILE" \
    --pred_path "$OUTPUT_FILE" \
    --embed_method "finetuned" \
    --scale "small"
