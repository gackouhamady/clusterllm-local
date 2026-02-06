#!/usr/bin/env bash
set -euo pipefail

# Absolute path to this script directory: .../triplet_prediction/scripts
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# triplet_prediction directory
TP_DIR="$(cd "${DIR}/.." && pwd)"

# repo root (scripts -> triplet_prediction -> perspective -> clusterllm -> src -> repo)
REPO_ROOT="$(cd "${DIR}/../../../../.." && pwd)"

# ---- Detect datasets directory (STRICT) ----
# Your repo has "data/" at root, sometimes "datasets/" exists in other repos.
# We try common locations, otherwise fail with a clear message.
if [ -d "${REPO_ROOT}/datasets" ]; then
  DATASETS_DIR="${REPO_ROOT}/datasets"
elif [ -d "${REPO_ROOT}/data/datasets" ]; then
  DATASETS_DIR="${REPO_ROOT}/data/datasets"
elif [ -d "${REPO_ROOT}/data" ]; then
  # fallback: if your datasets are directly under data/<dataset>/
  DATASETS_DIR="${REPO_ROOT}/data"
else
  echo "[ERROR] Cannot find datasets directory."
  echo "Tried: ${REPO_ROOT}/datasets, ${REPO_ROOT}/data/datasets, ${REPO_ROOT}/data"
  echo "Please create one of these or set DATASETS_DIR manually."
  exit 1
fi

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] DATASETS_DIR=${DATASETS_DIR}"

# ===== instructor-large =====
scale=small
for dataset in banking77 few_rel_nat stackexchange go_emotion
do
  for max_query in 1024
  do
    for embed in instructor
    do
      feat_path="${DATASETS_DIR}/${dataset}/${scale}_embeds.hdf5"
      data_path="${DATASETS_DIR}/${dataset}/${scale}.jsonl"

      if [ ! -f "${feat_path}" ]; then
        echo "[ERROR] Missing feat_path: ${feat_path}"
        exit 1
      fi
      if [ ! -f "${data_path}" ]; then
        echo "[ERROR] Missing data_path: ${data_path}"
        exit 1
      fi

      python "${TP_DIR}/random_triplet_sampling.py" \
        --data_path "${data_path}" \
        --feat_path "${feat_path}" \
        --dataset "${dataset}" \
        --embed_method "${embed}" \
        --max_query "${max_query}" \
        --filter_first_prop 0.0 \
        --large_ent_prop 0.2 \
        --out_dir "${TP_DIR}/sampled_triplet_results" \
        --max_distance 67 \
        --scale "${scale}" \
        --shuffle_inds \
        --seed 100
    done
  done
done

# ===== e5-large =====
# (same structure; enable if needed)
