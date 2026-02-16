#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/hamadygackou777/clusterllm-local"
GRAN_DIR="${REPO_ROOT}/src/clusterllm/granularity"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

cd "$GRAN_DIR"

echo "================================================================"
echo "🚀 LANCEMENT PREDICT_PAIRS (OLLAMA LOCAL)"
echo "================================================================"

# Use the recognized nickname from the ValueError list
MODEL_KEY="llama3_q4km"
INPUT_FILE="sampled_pair_results/banking77_embed=finetuned_s=small_k=1_multigran2-200_seed=100.json"
PROMPT_FILE="prompts_pair_exps_pair_v8.json"

# Note: The argument name must match what add_ollama_cli_args expects
# Usually --ollama-model
python predict_pairs.py \
    --dataset "banking77" \
    --data_path "$INPUT_FILE" \
    --prompt_file "$PROMPT_FILE" \
    --ollama-model "$MODEL_KEY" \
    --overwrite \
    --delay 0.1 \
    --save_every 10

echo "✅ Fin du script."