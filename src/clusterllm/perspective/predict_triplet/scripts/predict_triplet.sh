#!/usr/bin/env bash
set -euo pipefail

# 1. Trouve la racine du projet
REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"

# 2. Ajoute src au PYTHONPATH
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 3. Se déplace dans le dossier de travail
WORK_DIR="${REPO_ROOT}/src/clusterllm/perspective/predict_triplet"
echo "Déplacement vers : $WORK_DIR"
cd "$WORK_DIR"

# 4. Variables
DATASET="banking77"
# Fichier d'entrée (généré à l'étape précédente)
DATA_PATH="${WORK_DIR}/sampled_triplet_results/banking77_embed=instructor_s=small_m=500_d=67_choice_seed=42.json"
OLLAMA_MODEL="llama3_q4km"

# CORRECTION CRITIQUE : Création du dossier de sortie AVANT de lancer Python
OUT_DIR="predicted_triplet_results"
mkdir -p "$OUT_DIR"

echo "----------------------------------------------------------------"
echo "Prédiction des triplets"
echo "Entrée     : $DATA_PATH"
echo "Sortie dir : $WORK_DIR/$OUT_DIR"
echo "Modèle     : $OLLAMA_MODEL"
echo "----------------------------------------------------------------"

# 5. Exécution
# Note : on lance python depuis WORK_DIR pour qu'il trouve prompts.json
python predict.py \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --ollama-base-url "http://localhost:11434" \
    --ollama-model "$OLLAMA_MODEL" \
    --ollama-num-predict 64 \
    --delay 0

