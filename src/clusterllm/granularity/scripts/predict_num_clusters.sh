#!/usr/bin/env bash
set -euo pipefail

# --- CONFIGURATION ---
REPO_ROOT="/home/hamadygackou777/clusterllm-local"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
cd "${REPO_ROOT}/src/clusterllm/granularity"

# --- STRATÉGIE DE DÉTECTION AUTOMATIQUE ---
# On cherche le résultat de l'étape précédente (predict_pairs)
# C'est ce fichier qui contient les "Yes/No" nécessaires pour compter les clusters.
SEARCH_DIR="${REPO_ROOT}/src/clusterllm/granularity/predicted_pair_results"
INPUT_FILE=$(ls -t "${SEARCH_DIR}"/*.json 2>/dev/null | head -n 1)

if [ -z "$INPUT_FILE" ]; then
    echo "❌ ERREUR : Aucun fichier trouvé dans predicted_pair_results."
    echo "   Avez-vous lancé l'étape précédente (predict_pairs.sh) ?"
    exit 1
fi

echo "================================================================"
echo "📊 PRÉDICTION DU NOMBRE DE CLUSTERS"
echo "   Entrée détectée : $(basename "$INPUT_FILE")"
echo "================================================================"

# --- DÉFINITION DE LA SORTIE ---
OUT_DIR="${REPO_ROOT}/src/clusterllm/granularity/predicted_num_clusters_results"
mkdir -p "$OUT_DIR"
# On garde le même nom de fichier mais dans le dossier final
OUTPUT_FILE="${OUT_DIR}/clusters_$(basename "$INPUT_FILE")"

# --- LANCEMENT AVEC ARGUMENTS OFFICIELS ---
# --clustering_results : Le fichier JSON avec les prédictions Yes/No
# --pred_path          : Le fichier JSON final qui contiendra le nombre k
# --data_path          : Le fichier brut test.jsonl (pour avoir les nœuds)

python3 predict_num_clusters.py \
    --dataset "banking77" \
    --data_path "${REPO_ROOT}/datasets/banking77/test.jsonl" \
    --clustering_results "$INPUT_FILE" \
    --pred_path "$OUTPUT_FILE" \
    --embed_method "finetuned" \
    --scale "small"

echo "✅ Terminé. Résultat final :"
echo "   $OUTPUT_FILE"
