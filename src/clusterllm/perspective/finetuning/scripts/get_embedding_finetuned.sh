#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/home/hamadygackou777/clusterllm-local/src:${PYTHONPATH:-}"
cd "/home/hamadygackou777/clusterllm-local/src/clusterllm/perspective/finetuning"
python get_embedding.py     --task_name "banking77"     --data_path "/home/hamadygackou777/clusterllm-local/datasets/banking77/test.jsonl"     --result_file "/home/hamadygackou777/clusterllm-local/datasets/banking77/embeddings_finetuned.h5"     --model_name "/home/hamadygackou777/clusterllm-local/src/clusterllm/perspective/finetuning/checkpoints/banking77_finetuned"     --batch_size 32     --overwrite
