#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Repo root (robust)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

WORK_DIR="${REPO_ROOT}/src/clusterllm/perspective/finetuning"
cd "$WORK_DIR"

usage() {
  echo "Usage:"
  echo "  $0 --dataset <DATASET> --llm_source <LLM_TAG> --run_dir <RUN_DIR> [--train_kind standard|self] [--train_file FILE]"
  echo "     [--model_to_finetune HF_MODEL] [--output_dir DIR]"
  echo "     [--num_train_epochs N] [--per_device_train_batch_size B] [--learning_rate LR] [--cl_temperature T]"
  echo ""
  echo "Examples:"
  echo "  $0 --dataset bank77 --llm_source deepseek-r1_32b --run_dir runs/bank77_small/deepseek_finetuning --train_kind standard"
  echo "  $0 --dataset bank77 --llm_source deepseek-r1_32b --run_dir runs/bank77_small/deepseek_finetuning --train_kind self"
  exit 1
}

DATASET=""
LLM_SOURCE=""
RUN_DIR=""
TRAIN_KIND="standard"
TRAIN_FILE=""
MODEL_TO_FINETUNE="hkunlp/instructor-large"
OUTPUT_DIR=""

NUM_EPOCHS=3
BATCH_SIZE=2
LR="2e-5"
TEMP="0.05"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --llm_source) LLM_SOURCE="$2"; shift 2;;
    --run_dir) RUN_DIR="$2"; shift 2;;
    --train_kind) TRAIN_KIND="$2"; shift 2;;
    --train_file) TRAIN_FILE="$2"; shift 2;;
    --model_to_finetune) MODEL_TO_FINETUNE="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --num_train_epochs) NUM_EPOCHS="$2"; shift 2;;
    --per_device_train_batch_size) BATCH_SIZE="$2"; shift 2;;
    --learning_rate) LR="$2"; shift 2;;
    --cl_temperature) TEMP="$2"; shift 2;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

# MUST have these
[[ -z "$DATASET" || -z "$LLM_SOURCE" || -z "$RUN_DIR" ]] && usage

# If run_dir is relative, make it absolute
if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="${REPO_ROOT}/${RUN_DIR}"
fi

# Default finetuning_data dir (matches your pipeline)
FINETUNE_DATA_DIR="${RUN_DIR}/perspective/finetuning_data"
mkdir -p "$FINETUNE_DATA_DIR"

# Determine train file
if [[ -z "$TRAIN_FILE" ]]; then
  if [[ "$TRAIN_KIND" == "standard" ]]; then
    TRAIN_FILE="${FINETUNE_DATA_DIR}/triplets-${LLM_SOURCE}-train.json"
  elif [[ "$TRAIN_KIND" == "self" ]]; then
    # Your converter produced: triplets-self-train.json
    TRAIN_FILE="${FINETUNE_DATA_DIR}/triplets-self-train.json"
  else
    echo "❌ train_kind must be 'standard' or 'self' (got: $TRAIN_KIND)"
    exit 1
  fi
fi

# Default output dir
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${RUN_DIR}/perspective/checkpoints/${DATASET}_finetuned_instructor_${LLM_SOURCE}_${TRAIN_KIND}"
fi
mkdir -p "$OUTPUT_DIR"

# Safety checks
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "❌ ERREUR : Le fichier d'entraînement est introuvable !"
  echo "   TRAIN_FILE : $TRAIN_FILE"
  echo "   Vérifie que tu as bien lancé la conversion ($TRAIN_KIND) pour $LLM_SOURCE."
  exit 1
fi

echo "----------------------------------------------------------------"
echo "Démarrage du Finetuning (paper-style)"
echo "DATASET           : $DATASET"
echo "LLM_SOURCE        : $LLM_SOURCE"
echo "TRAIN_KIND        : $TRAIN_KIND"
echo "MODEL_TO_FINETUNE : $MODEL_TO_FINETUNE"
echo "TRAIN_FILE        : $TRAIN_FILE"
echo "OUTPUT_DIR        : $OUTPUT_DIR"
echo "EPOCHS            : $NUM_EPOCHS"
echo "BATCH_SIZE        : $BATCH_SIZE"
echo "LR                : $LR"
echo "CL_TEMPERATURE    : $TEMP"
echo "----------------------------------------------------------------"

python finetune.py \
  --model_name_or_path "$MODEL_TO_FINETUNE" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --do_train \
  --overwrite_output_dir \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --learning_rate "$LR" \
  --cl_temperature "$TEMP" \
  --remove_unused_columns False
