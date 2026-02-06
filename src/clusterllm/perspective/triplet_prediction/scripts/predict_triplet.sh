#!/usr/bin/env bash
set -euo pipefail

# --------------------------
# Local Ollama configuration
# --------------------------
# You can override these at runtime:
#   OLLAMA_MODEL="mistral:7b-instruct-q4_K_M" ./predict_triplet.sh
#   OLLAMA_MODEL="llama3:8b-instruct-q4_K_M"  ./predict_triplet.sh
#
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-mistral:7b-instruct-q4_K_M}"
OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-120}"
OLLAMA_TEMPERATURE="${OLLAMA_TEMPERATURE:-0.0}"
OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT:-64}"

# Optional: also export env vars that your factory/tools may read
export CLUSTERLLM_OLLAMA_BASE_URL="${OLLAMA_BASE_URL}"
export CLUSTERLLM_OLLAMA_MODEL_KEY="${OLLAMA_MODEL}"
export CLUSTERLLM_OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT}"
export CLUSTERLLM_OLLAMA_TEMPERATURE="${OLLAMA_TEMPERATURE}"
export CLUSTERLLM_OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT}"

for dataset in banking77
do
    link_path="sampled_triplet_results/${dataset}_embed=instructor_s=small_m=1024_d=67.0_sf_choice_seed=100.json"
    # link_path="sampled_triplet_results/${dataset}_embed=instructor_s=large_m=1024_d=67.0_sf_choice_seed=100.json"

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python ../predict.py \
        --dataset "${dataset}" \
        --data_path "${link_path}" \
        --delay 1 \
        --max_trials 10 \
        --save_every 50 \
        --ollama-base-url "${OLLAMA_BASE_URL}" \
        --ollama-model "${OLLAMA_MODEL}" \
        --ollama-timeout "${OLLAMA_TIMEOUT}" \
        --ollama-temperature "${OLLAMA_TEMPERATURE}" \
        --ollama-num-predict "${OLLAMA_NUM_PREDICT}"
done
