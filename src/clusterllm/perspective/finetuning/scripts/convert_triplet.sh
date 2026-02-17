#!/usr/bin/env bash
set -euo pipefail

# 1. Racine du projet
REPO_ROOT="/home/hamadygackou777/clusterllm-local"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 2. Dossier de travail
WORK_DIR="${REPO_ROOT}/src/clusterllm/perspective/finetuning"
echo "Déplacement vers : $WORK_DIR"
cd "$WORK_DIR"

# 3. Variables & Gestion du Modèle
DATASET="banking77"

# ==> NOUVEAUTÉ : On récupère l'argument 1 (ex: "qwen2.5:7b")
# Si aucun argument n'est donné, on garde "llama3_q4km" par défaut
MODEL_NAME="${1:-llama3_q4km}"

echo "----------------------------------------------------------------"
echo "Conversion (Standard) - Modèle : $MODEL_NAME"
echo "----------------------------------------------------------------"

# Construction du chemin avec le nom du modèle dynamique
PRED_FILE="${REPO_ROOT}/src/clusterllm/perspective/predict_triplet/predicted_triplet_results/banking77_embed=instructor_s=small_m=500_d=67_choice_seed=42-${MODEL_NAME}-pred.json"

DATA_PATH="${REPO_ROOT}/datasets/banking77/test.jsonl"
OUT_DIR="${WORK_DIR}/converted_triplet_results"

mkdir -p "$OUT_DIR"

# Vérification de sécurité
if [ ! -f "$PRED_FILE" ]; then
    echo "❌ ERREUR : Le fichier de prédiction est introuvable !"
    echo "   Chemin cherché : $PRED_FILE"
    echo "   Astuce : Avez-vous bien lancé l'étape précédente avec le modèle '$MODEL_NAME' ?"
    exit 1
fi

# 4. Exécution
python convert_triplet.py \
    --dataset "$DATASET" \
    --pred_path "$PRED_FILE" \
    --data_path "$DATA_PATH" \
    --output_path "$OUT_DIR"

echo "✅ Conversion standard terminée pour $MODEL_NAME."