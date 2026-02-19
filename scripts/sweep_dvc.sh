#!/usr/bin/env bash
set -euo pipefail

# Modèles actifs (tu peux aussi les lire depuis params.yaml si tu veux, mais là c’est simple et robuste)
LLMS=("llama3:latest" "qwen2.5:32b")

DATASETS=("bank77" "clinc_intent" "mtop_intent" "mtop_domain" "massive_intent" "massive_domain" "stackex" "arxiv" "reddit" "go_emotions")

slugify() {
  echo "$1" | tr ':.' '__' | tr -cd 'A-Za-z0-9_-'
}

for ds in "${DATASETS[@]}"; do
  for llm in "${LLMS[@]}"; do
    slug="$(slugify "$llm")"
    run_dir="runs/${ds}/${slug}"

    echo "=== RUN dataset=${ds} llm=${llm} run_dir=${run_dir} ==="
    dvc exp run \
      -S run.dataset="${ds}" \
      -S run.split_train="large" \
      -S run.split_eval="small" \
      -S run.ollama_model="${llm}" \
      -S run.run_dir="${run_dir}"
  done
done
