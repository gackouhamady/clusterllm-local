#!/usr/bin/env bash
set -euo pipefail

# 1. Project Root and Environment
REPO_ROOT="/home/hamadygackou777/clusterllm-local"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 2. Assign Arguments (Dynamic)
# $1: Nom du modèle (ex: qwen2.5:7b, default: qwen)
# $2: Nom du Dataset (ex: banking77, default: banking77)
MODEL_NAME="${1:-"qwen"}"
TASK_NAME="${2:-"banking77"}"

# 3. Working Directory
cd "${REPO_ROOT}/src/clusterllm/granularity"

echo "================================================================"
echo "🚀 GENERATING PAIRS FOR GRANULARITY ANALYSIS"
echo "Dataset : $TASK_NAME"
echo "Model   : $MODEL_NAME"
echo "================================================================"

# 4. Variables - Dynamic Paths
DATA_PATH="${REPO_ROOT}/datasets/${TASK_NAME}/test.jsonl"

# Fichier d'entrée : Vos embeddings finetunés
# Assurez-vous que ce fichier existe (celui généré à l'étape précédente)
FEAT_PATH="${REPO_ROOT}/datasets/${TASK_NAME}/embeddings_finetuned.h5"

# Dossier de sortie : On inclut le nom du modèle pour organiser les résultats
OUT_DIR="sampled_pair_results/${TASK_NAME}_${MODEL_NAME}"

# Safety Check
if [ ! -f "$FEAT_PATH" ]; then
    echo "❌ ERREUR : Fichier d'embeddings introuvable : $FEAT_PATH"
    echo "   Avez-vous lancé l'étape 'get_embedding.sh' avant ?"
    exit 1
fi

# 5. Execution
python sample_pairs.py \
    --dataset "$TASK_NAME" \
    --data_path "$DATA_PATH" \
    --feat_path "$FEAT_PATH" \
    --embed_method "finetuned" \
    --scale "small" \
    --k 3 \
    --min_clusters 2 \
    --max_clusters 200 \
    --out_dir "$OUT_DIR" \
    --seed 100

echo "✅ Pair sampling complete."
echo "📁 Results saved in : granularity/$OUT_DIR"