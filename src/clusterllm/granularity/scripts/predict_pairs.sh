#!/usr/bin/env bash
set -euo pipefail

# --- CONFIGURATION ---
REPO_ROOT="/home/hamadygackou777/clusterllm-local"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
cd "${REPO_ROOT}/src/clusterllm/granularity"

# --- CHEMINS ---
INPUT_FILE="${REPO_ROOT}/src/clusterllm/granularity/sampled_pair_results/banking77_embed=finetuned_s=small_k=1_multigran2-200_seed=100.json"
PROMPT_FILE="${REPO_ROOT}/src/clusterllm/granularity/prompts/pair_prediction.json"

echo "================================================================"
echo "🚀 LANCEMENT PREDICT_PAIRS (OLLAMA LOCAL)"
echo "================================================================"

# NOTE : J'ai retiré --output_dir qui causait l'erreur.
# Le script sauvegardera automatiquement dans 'predicted_pair_results'.

python3 predict_pairs.py \
    --dataset "banking77" \
    --data_path "$INPUT_FILE" \
    --prompt_file "$PROMPT_FILE" \
    --temperature 0.0 \
    --ollama-model "mistral_q4km" \
    --ollama-base-url "http://localhost:11434"

echo "✅ Fin du script."
