#!/usr/bin/env bash
set -euo pipefail

# 1. Racine du projet
REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 2. Dossier de travail
WORK_DIR="${REPO_ROOT}/src/clusterllm/perspective/finetuning"
echo "Déplacement vers : $WORK_DIR"
cd "$WORK_DIR"

# 3. Variables
DATASET="banking77"
# Le fichier JSON généré par Ollama
PRED_FILE="${REPO_ROOT}/src/clusterllm/perspective/triplet_prediction/predicted_triplet_results/banking77_embed=instructor_s=small_m=500_d=67_choice_seed=42-llama3_q4km-pred.json"
# Le fichier original des données
DATA_PATH="${REPO_ROOT}/datasets/banking77/test.jsonl"
OUT_DIR="${WORK_DIR}/converted_triplet_results"

mkdir -p "$OUT_DIR"

echo "----------------------------------------------------------------"
echo "Conversion (Standard)"
echo "----------------------------------------------------------------"

# 4. Exécution avec les arguments CORRIGÉS (--pred_path, --output_path)
python convert_triplet.py \
    --dataset "$DATASET" \
    --pred_path "$PRED_FILE" \
    --data_path "$DATA_PATH" \
    --output_path "$OUT_DIR"

