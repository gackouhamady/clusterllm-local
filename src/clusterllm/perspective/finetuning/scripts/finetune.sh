#!/usr/bin/env bash
set -euo pipefail

# 1. Memory Optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Project Paths
REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 3. Work Directory
WORK_DIR="${REPO_ROOT}/src/clusterllm/perspective/finetuning"
echo "Moving to: $WORK_DIR"
cd "$WORK_DIR"

# 4. Variables
TRAIN_FILE="${WORK_DIR}/converted_triplet_results/banking77_embed=instructor_s=small_m=500_d=67_choice_seed=42-llama3_q4km-train.json"

# COHERENT MODEL CHOICE
MODEL_NAME="hkunlp/instructor-large"
OUTPUT_DIR="${WORK_DIR}/checkpoints/banking77_finetuned_instructor"

mkdir -p "$OUTPUT_DIR"

echo "----------------------------------------------------------------"
echo "Starting COHERENT Finetuning (Instructor-Large)"
echo "----------------------------------------------------------------"

# 5. Execution
# We removed --query_instruction because your script doesn't support the flag.
# We keep batch size at 2 to avoid OOM on the L4.
python finetune.py \
    --train_file "$TRAIN_FILE" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-5 \
    --cl_temperature 0.05