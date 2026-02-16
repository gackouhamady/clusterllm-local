#!/usr/bin/env bash
set -euo pipefail

# 1. On trouve où se situe ce script .sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 2. Le dossier parent (finetuning)
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# 3. La racine du projet (clusterllm-local)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

TASK_NAME="banking77"
DATA_PATH="${REPO_ROOT}/datasets/banking77/test.jsonl"
RESULT_FILE="${REPO_ROOT}/datasets/banking77/embeddings.hdf5"
MODEL_NAME="hkunlp/instructor-large"

# On lance le python en utilisant le chemin complet vers le fichier .py
python "${PARENT_DIR}/get_embedding.py" \
    --task_name "$TASK_NAME" \
    --data_path "$DATA_PATH" \
    --result_file "$RESULT_FILE" \
    --model_name "$MODEL_NAME" \
    --batch_size 8 \
    --measure \
    --overwrite