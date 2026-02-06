#!/usr/bin/env bash
set -euo pipefail

# Absolute path to this script directory: .../finetuning/scripts
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# finetuning directory
FT_DIR="$(cd "${DIR}/.." && pwd)"

# repo root (scripts -> finetuning -> perspective -> clusterllm -> src -> repo)
REPO_ROOT="$(cd "${DIR}/../../../../.." && pwd)"

# Detect datasets dir
if [ -d "${REPO_ROOT}/datasets" ]; then
  DATASETS_DIR="${REPO_ROOT}/datasets"
elif [ -d "${REPO_ROOT}/data/datasets" ]; then
  DATASETS_DIR="${REPO_ROOT}/data/datasets"
elif [ -d "${REPO_ROOT}/data" ]; then
  DATASETS_DIR="${REPO_ROOT}/data"
else
  echo "[ERROR] Cannot find datasets directory under ${REPO_ROOT}."
  exit 1
fi

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] DATASETS_DIR=${DATASETS_DIR}"

# ===== original embedding =====
for dataset in banking77
do
  for scale in small
  do
    data_path="${DATASETS_DIR}/${dataset}/${scale}.jsonl"
    result_file="${DATASETS_DIR}/${dataset}/${scale}_embeds.hdf5"

    if [ ! -f "${data_path}" ]; then
      echo "[ERROR] Missing data_path: ${data_path}"
      exit 1
    fi

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python "${FT_DIR}/get_embedding.py" \
      --model_name hkunlp/instructor-large \
      --scale "${scale}" \
      --task_name "${dataset}" \
      --data_path "${data_path}" \
      --result_file "${result_file}" \
      --measure
  done
done
