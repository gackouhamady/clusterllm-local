#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/hamadygackou777/clusterllm-local"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Dossier de travail
cd "${REPO_ROOT}/src/clusterllm/granularity"

echo "================================================================"
echo "🚀 LANCEMENT PREDICT_PAIRS AVEC FEW-SHOT PROMPTING"
echo "================================================================"

# Variables
INPUT_FILE="sampled_pair_results/banking77_embed=finetuned_s=small_k=1_multigran2-200_seed=100.json"
PROMPT_FILE="prompts_pair_exps_pair_v8.json"
MODEL="mistral:latest" # Vérifie bien le nom exact dans 'ollama list'

python predict.py \
    --input_path "$INPUT_FILE" \
    --prompt_path "$PROMPT_FILE" \
    --dataset "banking77" \
    --model "$MODEL" \
    --output_dir "predicted_pair_results" \
    --overwrite # Force le recalcul pour éviter la vitesse suspecte