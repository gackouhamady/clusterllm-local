#!/usr/bin/env bash
set -euo pipefail

# 1. Project Root and Environment
REPO_ROOT="/home/hamadygackou777/clusterllm-local"
GRAN_DIR="${REPO_ROOT}/src/clusterllm/granularity"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 2. Assign Arguments
MODEL_NAME="${1:-"qwen2.5:7b"}"
DATASET="${2:-"banking77"}"

# 3. Working Directory
cd "$GRAN_DIR"

# 4. Variables & Paths
prompt_path="prompts_pair_exps_pair_v8.json"
INPUT_DIR="sampled_pair_results/${DATASET}_${MODEL_NAME}"
# Dans scripts/prepare_prompts.sh, changez la ligne par :
sampled_pair_path="${INPUT_DIR}/${DATASET}_embed=finetuned_s=small_k=3_multigran2-200_seed=100.json"
data_path="${REPO_ROOT}/datasets/${DATASET}/test.jsonl"

echo "================================================================"
echo "🚀 PREPARING FEW-SHOT PROMPTS"
echo "Dataset : $DATASET"
echo "Model   : $MODEL_NAME"
echo "================================================================"

# Safety checks
if [ ! -f "$sampled_pair_path" ]; then
    echo "❌ ERROR: Sampled pair file not found at: $sampled_pair_path"
    exit 1
fi

# 5. Execution
python sample_pairs_for_prompt.py \
    --prompt_path "$prompt_path" \
    --sampled_pair_path "$sampled_pair_path" \
    --data_path "$data_path" \
    --dataset "$DATASET" \
    --num_sampled 16 \
    --num_for_prompt 2 \
    --seed 1234

echo "✅ Few-shot prompt preparation complete."
