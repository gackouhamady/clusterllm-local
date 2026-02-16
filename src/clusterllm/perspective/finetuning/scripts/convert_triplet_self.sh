#!/usr/bin/env bash
set -euo pipefail

# 1. Racine du projet (on remonte au dossier clusterllm-local)
REPO_ROOT="/home/hamadygackou777/clusterllm-local"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 2. Dossier de travail
WORK_DIR="${REPO_ROOT}/src/clusterllm/perspective/finetuning"
echo "Déplacement vers : $WORK_DIR"
cd "$WORK_DIR"

# 3. Variables
DATASET="banking77"

# --- CORRECTION DU CHEMIN ICI : predict_triplet au lieu de triplet_prediction ---
PRED_FILE="${REPO_ROOT}/src/clusterllm/perspective/predict_triplet/predicted_triplet_results/banking77_embed=instructor_s=small_m=500_d=67_choice_seed=42-llama3_q4km-pred.json"

DATA_PATH="${REPO_ROOT}/datasets/banking77/test.jsonl"
FEAT_PATH="${REPO_ROOT}/datasets/banking77/embeddings.pkl"
OUT_DIR="${WORK_DIR}/converted_triplet_results"

mkdir -p "$OUT_DIR"

# Vérification de sécurité pour le fichier de prédiction
if [ ! -f "$PRED_FILE" ]; then
    echo "❌ ERREUR : Le fichier de prédiction est introuvable à l'adresse suivante :"
    echo "$PRED_FILE"
    exit 1
fi

echo "----------------------------------------------------------------"
echo "Conversion (Self) - Utilisation du fichier LLM"
echo "----------------------------------------------------------------"

# 4. Exécution
python convert_triplet_self.py \
    --dataset "$DATASET" \
    --pred_path "$PRED_FILE" \
    --data_path "$DATA_PATH" \
    --feat_path "$FEAT_PATH" \
    --output_path "$OUT_DIR"

echo "✅ Conversion terminée. Fichiers disponibles dans $OUT_DIR"