#!/usr/bin/env bash
set -e

# --- CHEMINS ---
REPO_ROOT="$(pwd)"
CHECKPOINT_DIR="${REPO_ROOT}/src/clusterllm/perspective/finetuning/checkpoints/banking77_finetuned"
DATASET_DIR="${REPO_ROOT}/datasets/banking77"
GRANULARITY_DIR="${REPO_ROOT}/src/clusterllm/granularity"

echo "========================================================"
echo "🔧 CORRECTIF FINAL : FORMAT HDF5 (.h5)"
echo "========================================================"

# 1. GENERATION DES EMBEDDINGS (Extension .h5)
echo "[1/3] Génération des Embeddings (Format H5)..."
EMBED_SCRIPT="${REPO_ROOT}/src/clusterllm/perspective/finetuning/scripts/get_embedding_finetuned.sh"

cat > "$EMBED_SCRIPT" <<BASH
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH:-}"
cd "${REPO_ROOT}/src/clusterllm/perspective/finetuning"
python get_embedding.py \
    --task_name "banking77" \
    --data_path "${DATASET_DIR}/test.jsonl" \
    --result_file "${DATASET_DIR}/embeddings_finetuned.h5" \
    --model_name "${CHECKPOINT_DIR}" \
    --batch_size 32 \
    --overwrite
BASH
chmod +x "$EMBED_SCRIPT"
"$EMBED_SCRIPT"

# 2. ECHANTILLONNAGE (Lecture .h5)
echo "[2/3] Échantillonnage des paires (Lecture H5)..."
SAMPLE_SCRIPT="${GRANULARITY_DIR}/scripts/sample_pairs.sh"

cat > "$SAMPLE_SCRIPT" <<BASH
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH:-}"
cd "${GRANULARITY_DIR}"
# Note: On pointe bien vers le fichier .h5 généré juste avant
python sample_pairs.py \
    --dataset "banking77" \
    --data_path "${DATASET_DIR}/test.jsonl" \
    --feat_path "${DATASET_DIR}/embeddings_finetuned.h5" \
    --scale "small" \
    --embed_method "finetuned" \
    --k 1 \
    --out_dir "sampled_pair_results" \
    --min_clusters 2 \
    --max_clusters 200 \
    --seed 100
BASH
chmod +x "$SAMPLE_SCRIPT"
"$SAMPLE_SCRIPT"

# 3. VERIFICATION
RESULT_JSON="${GRANULARITY_DIR}/sampled_pair_results/banking77_embed=finetuned_s=small_k=1_multigran2-200_seed=100.json"
if [ -f "$RESULT_JSON" ]; then
    echo "========================================================"
    echo "🎉 SUCCÈS TOTAL : Paires générées !"
    echo "Fichier résultat : $RESULT_JSON"
    echo "Vous pouvez maintenant lancer predict_pairs.sh ou predict_num_clusters.sh"
    echo "========================================================"
else
    echo "❌ Erreur : Le fichier JSON final n'est pas là."
    exit 1
fi
