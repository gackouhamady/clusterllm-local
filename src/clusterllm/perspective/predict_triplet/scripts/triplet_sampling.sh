#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Configuration
DATASET="${DATASET:-bank77}"
SPLIT="${SPLIT:-small}"
SEED="${SEED:-42}"
NUM_QUERIES="${NUM_QUERIES:-1024}"

# DOSSIERS : On sépare bien l'entrée de la sortie
DATA_PATH="${REPO_ROOT}/src/clusterllm/datasets/${DATASET}/${SPLIT}.jsonl"

# Utilise l'embedding déjà généré (vérifie les deux emplacements possibles)
EMBED_PATH="${INIT_EMB_PATH:-${REPO_ROOT}/runs/test_manual/${DATASET}_${SPLIT}_embeds.hdf5}"

# Dossier où stocker les triplets pour DeepSeek
OUT_DIR="${REPO_ROOT}/runs/${DATASET}_${SPLIT}/deepseek_finetuning/triplets"

if [[ ! -f "$EMBED_PATH" ]]; then
    echo "❌ Erreur: Embedding introuvable à $EMBED_PATH"
    echo "   Lancez 'export INIT_EMB_PATH=/votre/chemin/file.hdf5' si le fichier est ailleurs."
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "----------------------------------------------------------------"
echo "CLUSTERLLM - Sampling Triplets for Perspective Improvement"
echo "Input Embedding : $EMBED_PATH"
echo "Output Directory: $OUT_DIR"
echo "----------------------------------------------------------------"

poetry run python "${REPO_ROOT}/src/clusterllm/perspective/predict_triplet/triplet_sampling.py" \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --embed_path "$EMBED_PATH" \
    --output_dir "$OUT_DIR" \
    --num_queries "$NUM_QUERIES" \
    --scale "$SPLIT" \
    --seed "$SEED"