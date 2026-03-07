#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "dvc.yaml" ]]; then
  echo "ERREUR: dvc.yaml introuvable dans ${REPO_ROOT}" >&2
  exit 1
fi

if [[ ! -f "params.yaml" ]]; then
  echo "ERREUR: params.yaml introuvable dans ${REPO_ROOT} (il doit contenir run: et run2llms:)" >&2
  exit 1
fi

DATASETS=(
  bank77
  clinc150
  massive_intent
  massive_domain
  mtop_intent
  mtop_domain
  go_emotions
  reddit
  stackex
  arxiv
  few_event
  few_nerd_nat
  few_rel_nat
  clinc_intent
  clinc_domain
)

LLM_TRIPLET='qwen2.5:3b'
LLM_PAIRS='llama3.3:70b-instruct-q2_K'
SEED='42'

run_real () {
  local ds="$1" sc="$2"
  poetry run dvc exp run full_pipeline_2llms \
    -S "run.dataset=${ds}" \
    -S "run.scale=${sc}" \
    -S "run.llm_triplet=${LLM_TRIPLET}" \
    -S "run.llm_pairs=${LLM_PAIRS}" \
    -S "run.seed=${SEED}"
}

run_cont () {
  local ds="$1" sc="$2"
  poetry run dvc exp run full_pipeline_2llms_cont \
    -S "run2llms.dataset=${ds}" \
    -S "run2llms.scale=${sc}" \
    -S "run2llms.llm_triplet=${LLM_TRIPLET}" \
    -S "run2llms.llm_pairs=${LLM_PAIRS}" \
    -S "run2llms.seed=${SEED}"
}

echo "=== SMALL: real then cont ==="
for ds in "${DATASETS[@]}"; do
  echo "--- REAL small: ${ds}"
  run_real "$ds" "small"
done
for ds in "${DATASETS[@]}"; do
  echo "--- CONT small: ${ds}"
  run_cont "$ds" "small"
done

echo "=== LARGE: real then cont ==="
for ds in "${DATASETS[@]}"; do
  echo "--- REAL large: ${ds}"
  run_real "$ds" "large"
done
for ds in "${DATASETS[@]}"; do
  echo "--- CONT large: ${ds}"
  run_cont "$ds" "large"
done

echo "OK ✅ all done"
