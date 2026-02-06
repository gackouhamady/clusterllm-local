#!/usr/bin/env bash
set -euo pipefail

# 1. Trouve la racine du projet
REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"

# 2. Ajoute src au PYTHONPATH
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 3. Se déplace dans le dossier finetuning
WORK_DIR="${REPO_ROOT}/src/clusterllm/perspective/finetuning"
echo "Déplacement vers : $WORK_DIR"
cd "$WORK_DIR"

# 4. Variables
TASK_NAME="banking77"
DATA_PATH="${REPO_ROOT}/datasets/banking77/test.jsonl"
RESULT_FILE="${REPO_ROOT}/datasets/banking77/embeddings.pkl"
# CORRECTION : On définit un modèle précis pour éviter l'erreur "None"
MODEL_NAME="sentence-transformers/all-mpnet-base-v2"

echo "----------------------------------------------------------------"
echo "Tâche      : $TASK_NAME"
echo "Entrée     : $DATA_PATH"
echo "Modèle     : $MODEL_NAME"
echo "Sortie     : $RESULT_FILE"
echo "----------------------------------------------------------------"

# 5. Exécution avec l'argument --model_name AJOUTÉ
python get_embedding.py \
    --task_name "$TASK_NAME" \
    --data_path "$DATA_PATH" \
    --result_file "$RESULT_FILE" \
    --model_name "$MODEL_NAME" \
    --batch_size 32 \
    --overwrite

