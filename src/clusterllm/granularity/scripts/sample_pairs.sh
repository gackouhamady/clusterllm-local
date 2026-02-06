#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/home/hamadygackou777/clusterllm-local/src:${PYTHONPATH:-}"
cd "/home/hamadygackou777/clusterllm-local/src/clusterllm/granularity"
# Note: On pointe bien vers le fichier .h5 généré juste avant
python sample_pairs.py     --dataset "banking77"     --data_path "/home/hamadygackou777/clusterllm-local/datasets/banking77/test.jsonl"     --feat_path "/home/hamadygackou777/clusterllm-local/datasets/banking77/embeddings_finetuned.h5"     --scale "small"     --embed_method "finetuned"     --k 1     --out_dir "sampled_pair_results"     --min_clusters 2     --max_clusters 200     --seed 100
