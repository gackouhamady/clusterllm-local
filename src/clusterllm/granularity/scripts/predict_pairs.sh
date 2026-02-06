#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/home/hamadygackou777/clusterllm-local/src:${PYTHONPATH:-}"
cd "/home/hamadygackou777/clusterllm-local/src/clusterllm/granularity"

# Note : Si vous n'avez pas de clé OpenAI, vous pouvez simuler ou changer le model_name
# Ici on configure pour une exécution standard.
python predict_pairs.py     --input_file "/home/hamadygackou777/clusterllm-local/src/clusterllm/granularity/sampled_pair_results/banking77_embed=finetuned_s=small_k=1_multigran2-200_seed=100.json"     --output_dir "/home/hamadygackou777/clusterllm-local/src/clusterllm/granularity/predicted_pair_results"     --model_name "gpt-3.5-turbo"     --prompt_file "prompts/pair_prediction.txt"     --temperature 0.0
