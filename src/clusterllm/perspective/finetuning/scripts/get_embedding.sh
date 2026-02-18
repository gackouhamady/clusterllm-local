#!/usr/bin/env bash
set -euo pipefail

# DVC-friendly: fail fast if missing
: "${DATASET:?Missing DATASET}"
: "${SPLIT_TRAIN:?Missing SPLIT_TRAIN}"
: "${RUN_DIR:?Missing RUN_DIR}"

# Optional (your dvc stage already sets CUDA_VISIBLE_DEVICES)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Resolve repo root from this script location (robust with/without DVC wdir)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Work inside finetuning dir (same as official repo style)
WORK_DIR="${REPO_ROOT}/src/clusterllm/perspective/finetuning"
cd "${WORK_DIR}"

# Map your DVC split -> official "scale"
# You can adjust this mapping if your project uses different scale naming.
SCALE="${SCALE:-${SPLIT_TRAIN}}"

# Input: prefer your DVC-prepared CSV/JSONL if you have it.
# If you truly want the official path, keep it as datasets/<dataset>/<scale>.jsonl
# Here we keep the official structure but allow overriding.
DATA_PATH_DEFAULT="${WORK_DIR}/../../datasets/${DATASET}/${SCALE}.jsonl"
DATA_PATH="${DATA_PATH:-${DATA_PATH_DEFAULT}}"

# Output: MUST match your DVC outs: ${run.run_dir}/perspective/emb_init
OUT_DIR="${REPO_ROOT}/${RUN_DIR}/perspective/emb_init"
mkdir -p "${OUT_DIR}"
RESULT_FILE="${OUT_DIR}/${SCALE}_embeds.hdf5"

MODEL_NAME="${MODEL_NAME:-hkunlp/instructor-large}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/.cache/hf}"

echo "----------------------------------------------------------------"
echo "Embedding extraction (official-style, DVC adapted)"
echo "DATASET     : ${DATASET}"
echo "SCALE       : ${SCALE}"
echo "DATA_PATH   : ${DATA_PATH}"
echo "RESULT_FILE : ${RESULT_FILE}"
echo "MODEL_NAME  : ${MODEL_NAME}"
echo "RUN_DIR     : ${RUN_DIR}"
echo "----------------------------------------------------------------"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python get_embedding.py \
  --model_name "${MODEL_NAME}" \
  --scale "${SCALE}" \
  --task_name "${DATASET}" \
  --data_path "${DATA_PATH}" \
  --cache_dir "${CACHE_DIR}" \
  --result_file "${RESULT_FILE}" \
  --measure \
  --overwrite
