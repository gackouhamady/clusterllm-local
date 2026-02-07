#!/bin/bash
export PYTHONPATH="/home/hamadygackou777/clusterllm-local/src"
python get_embedding.py     --task_name "banking77"     --data_path "/home/hamadygackou777/clusterllm-local/datasets/banking77/test.jsonl"     --result_file "/home/hamadygackou777/clusterllm-local/datasets/banking77/embeddings.pkl"     --model_name "sentence-transformers/all-mpnet-base-v2"     --batch_size 32     --overwrite
