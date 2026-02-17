#!/usr/bin/env bash
set -euo pipefail

# 1. Environnement
REPO_ROOT="/home/hamadygackou777/clusterllm-local"
GRAN_DIR="${REPO_ROOT}/src/clusterllm/granularity"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 2. Arguments
MODEL_NAME="${1:-"qwen2.5:7b"}"
DATASET="${2:-"banking77"}"

cd "$GRAN_DIR"

echo "================================================================"
echo "🚀 LANCEMENT PREDICT_PAIRS (OLLAMA LOCAL)"
echo "Dataset : $DATASET"
echo "Model   : $MODEL_NAME"
echo "================================================================"

# 3. Chemins dynamiques
INPUT_DIR="sampled_pair_results/${DATASET}_${MODEL_NAME}"
# Dans scripts/predict_pairs.sh, changez la ligne par :
INPUT_FILE="${INPUT_DIR}/${DATASET}_embed=finetuned_s=small_k=3_multigran2-200_seed=100.json"
PROMPT_FILE="prompts_pair_exps_pair_v8.json"

# 4. Exécution
python predict_pairs.py \
    --dataset "$DATASET" \
    --data_path "$INPUT_FILE" \
    --prompt_file "$PROMPT_FILE" \
    --ollama-model "$MODEL_NAME" \
    --overwrite \
    --delay 0.05 \
    --save_every 10

echo "✅ Prédictions terminées."
