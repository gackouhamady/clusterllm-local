#!/bin/bash
set -euo pipefail

# 1. Récupération des variables DVC
# On utilise des valeurs par défaut si les variables ne sont pas définies (sécurité)
MODEL=${OLLAMA_MODEL:-"llama3:8b-instruct-q4_K_M"}
BASE_RUN_DIR=${RUN_DIR:-"runs/debug"}

# 2. Définition des chemins d'Entrée et de Sortie
# C'est ici qu'on redirige le script vers les dossiers gérés par DVC
INPUT_DIR="${BASE_RUN_DIR}/perspective/triplets_sampled"
OUTPUT_DIR="${BASE_RUN_DIR}/perspective/triplets_predicted"

# Création du dossier de sortie (sécurité)
mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo " CLUSTERLLM - PREDICTION DES TRIPLETS"
echo "========================================================"
echo "Dataset     : $DATASET"
echo "Modèle LLM  : $MODEL"
echo "Input Dir   : $INPUT_DIR"
echo "Output Dir  : $OUTPUT_DIR"
echo "========================================================"

# 3. Exécution du script Python avec les bons arguments
# On force le script python à lire et écrire aux bons endroits
python predict.py \
    --dataset "$DATASET" \
    --model_name "$MODEL" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --seed "${SEED:-42}"

echo "--> Prédiction terminée."