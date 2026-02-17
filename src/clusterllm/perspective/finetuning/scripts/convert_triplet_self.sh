#!/usr/bin/env bash
set -euo pipefail

# 1. Racine du projet (Dynamique pour être plus sûr, ou votre chemin en dur)
REPO_ROOT="/home/hamadygackou777/clusterllm-local"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 2. Dossier de travail
WORK_DIR="${REPO_ROOT}/src/clusterllm/perspective/finetuning"
echo "Déplacement vers : $WORK_DIR"
cd "$WORK_DIR"

# 3. Variables & Gestion du Modèle
DATASET="banking77"

# ==> ICI : On récupère le nom du modèle depuis la commande (comme tout à l'heure)
# Par défaut : "llama3_q4km" si rien n'est précisé
MODEL_NAME="${1:-llama3_q4km}"

echo "----------------------------------------------------------------"
echo "Conversion (Self) - Recherche des résultats pour : $MODEL_NAME"
echo "----------------------------------------------------------------"

# Construction du nom de fichier basé sur le modèle
# ATTENTION : Si vous avez passé "qwen2.5:7b", vérifiez que le fichier généré contient bien "qwen2.5:7b" ou "qwen2.5_7b"
PRED_FILE="${REPO_ROOT}/src/clusterllm/perspective/predict_triplet/predicted_triplet_results/banking77_embed=instructor_s=small_m=500_d=67_choice_seed=42-${MODEL_NAME}-pred.json"

DATA_PATH="${REPO_ROOT}/datasets/banking77/test.jsonl"
FEAT_PATH="${REPO_ROOT}/datasets/banking77/embeddings.pkl"
OUT_DIR="${WORK_DIR}/converted_triplet_results"

mkdir -p "$OUT_DIR"

# Vérification de sécurité
if [ ! -f "$PRED_FILE" ]; then
    echo "❌ ERREUR : Le fichier de prédiction est introuvable !"
    echo "   Chemin cherché : $PRED_FILE"
    echo "   Avez-vous lancé l'étape précédente avec le modèle '$MODEL_NAME' ?"
    exit 1
fi

# 4. Exécution
python convert_triplet_self.py \
    --dataset "$DATASET" \
    --pred_path "$PRED_FILE" \
    --data_path "$DATA_PATH" \
    --feat_path "$FEAT_PATH" \
    --output_path "$OUT_DIR"

echo "✅ Conversion terminée. Fichiers disponibles dans $OUT_DIR"