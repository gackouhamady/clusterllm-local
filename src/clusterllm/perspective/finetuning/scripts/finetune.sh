#!/usr/bin/env bash
set -euo pipefail

# 1. Optimisation Mémoire & Gestion CUDA
# On active la gestion intelligente de la mémoire pour éviter les crashs OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Racine du projet
REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 3. Dossier de travail
WORK_DIR="${REPO_ROOT}/src/clusterllm/perspective/finetuning"
echo "Déplacement vers : $WORK_DIR"
cd "$WORK_DIR"

# 4. Variables
TRAIN_FILE="${WORK_DIR}/converted_triplet_results/banking77_embed=instructor_s=small_m=500_d=67_choice_seed=42-llama3_q4km-train.json"
MODEL_NAME="sentence-transformers/all-mpnet-base-v2"
OUTPUT_DIR="${WORK_DIR}/checkpoints/banking77_finetuned"

mkdir -p "$OUTPUT_DIR"

echo "----------------------------------------------------------------"
echo "Lancement du Finetuning (Mode Robuste)"
echo "----------------------------------------------------------------"

# 5. Exécution
# Ajout de --cl_temperature pour la clarté, même si le code est maintenant blindé
python finetune.py \
    --train_file "$TRAIN_FILE" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --cl_temperature 0.05

