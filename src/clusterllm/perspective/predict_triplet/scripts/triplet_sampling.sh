#!/usr/bin/env bash
set -euo pipefail

# 1. Trouve la racine du projet
REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"

# 2. Ajoute src au PYTHONPATH
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 3. Se déplace dans le dossier de travail
WORK_DIR="${REPO_ROOT}/src/clusterllm/perspective/predict_triplet"
echo "Déplacement vers : $WORK_DIR"
cd "$WORK_DIR"

# 4. Variables
DATASET="banking77"
DATA_PATH="${REPO_ROOT}/datasets/banking77/test.jsonl"
# CORRECTION 1: On utilise FEAT_PATH pour pointer vers les embeddings
FEAT_PATH="${REPO_ROOT}/datasets/banking77/embeddings.pkl"
# CORRECTION 2: On définit le dossier de sortie
OUT_DIR="${WORK_DIR}/sampled_triplet_results"

# Création du dossier de résultats
mkdir -p "$OUT_DIR"

echo "----------------------------------------------------------------"
echo "Échantillonnage (Sampling)"
echo "Embeddings : $FEAT_PATH"
echo "Sortie     : $OUT_DIR"
echo "----------------------------------------------------------------"

# 5. Exécution avec les arguments CORRIGÉS (--feat_path, --out_dir, --max_query)
python sampling.py \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --feat_path "$FEAT_PATH" \
    --out_dir "$OUT_DIR" \
    --max_query 500 \
    --seed 42

