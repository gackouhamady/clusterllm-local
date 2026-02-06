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
TRAIN_FILE="${WORK_DIR}/converted_triplet_results/banking77_embed=instructor_s=small_m=500_d=67_choice_seed=42-llama3_q4km-train.json"
MODEL_NAME="sentence-transformers/all-mpnet-base-v2"
OUTPUT_DIR="${WORK_DIR}/checkpoints/banking77_finetuned"

mkdir -p "$OUTPUT_DIR"

echo "----------------------------------------------------------------"
echo "Lancement du Finetuning (Stratégique)"
echo "----------------------------------------------------------------"

# 4. Exécution (Argument --max_seq_length RETIRÉ)
python finetune.py \
    --train_file "$TRAIN_FILE" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5

