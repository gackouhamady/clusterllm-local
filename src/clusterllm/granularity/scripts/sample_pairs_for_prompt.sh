#!/usr/bin/env bash
set -euo pipefail

# 1. Project Root and Environment
REPO_ROOT="/home/hamadygackou777/clusterllm-local"
GRAN_DIR="${REPO_ROOT}/src/clusterllm/granularity"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 2. Assign Arguments (Dynamic)
# $1: Nom du modèle (doit correspondre au nom utilisé dans sample_pairs.sh)
# $2: Nom du Dataset
MODEL_NAME="${1:-"qwen2.5:7b"}"
DATASET="${2:-"banking77"}"

# 3. Working Directory
cd "$GRAN_DIR"
echo "📍 Working directory: $PWD"

# 4. Variables & Paths
prompt_path="prompts_pair_exps_pair_v8.json"
# On pointe vers le dossier créé par le script sample_pairs.sh dynamique
INPUT_DIR="sampled_pair_results/${DATASET}_${MODEL_NAME}"
sampled_pair_path="${INPUT_DIR}/${DATASET}_embed=finetuned_s=small_k=3_multigran2-200_seed=100.json"
data_path="${REPO_ROOT}/datasets/${DATASET}/test.jsonl"

echo "----------------------------------------------------------------"
echo "🚀 PREPARING FEW-SHOT PROMPTS"
echo "Dataset : $DATASET"
echo "Model   : $MODEL_NAME"
echo "Source  : $sampled_pair_path"
echo "----------------------------------------------------------------"

# Safety checks
if [ ! -f "$sampled_pair_path" ]; then
    echo "❌ ERROR: Sampled pair file not found at: $sampled_pair_path"
    exit 1
fi

if [ ! -f "$data_path" ]; then
    echo "❌ ERROR: Data file not found at: $data_path"
    exit 1
fi

# 5. Execution
# On échantillonne 16 paires candidates pour en choisir 4 (2 Yes, 2 No) [cite: 277, 536]
python sample_pairs_for_prompt.py \
    --prompt_path "$prompt_path" \
    --sampled_pair_path "$sampled_pair_path" \
    --data_path "$data_path" \
    --dataset "$DATASET" \
    --num_sampled 16 \
    --num_for_prompt 2 \
    --seed 1234

echo "✅ Few-shot prompt preparation complete for $DATASET ($MODEL_NAME)."