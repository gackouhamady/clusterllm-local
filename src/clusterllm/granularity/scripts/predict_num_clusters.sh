#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/home/hamadygackou777/clusterllm-local/src:${PYTHONPATH:-}"
cd "/home/hamadygackou777/clusterllm-local/src/clusterllm/granularity"

python predict_num_clusters.py     --input_file "/home/hamadygackou777/clusterllm-local/src/clusterllm/granularity/sampled_pair_results/banking77_embed=finetuned_s=small_k=1_multigran2-200_seed=100.json"     --output_dir "/home/hamadygackou777/clusterllm-local/src/clusterllm/granularity/predicted_num_clusters_results"
