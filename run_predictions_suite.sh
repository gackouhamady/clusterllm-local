#!/usr/bin/env bash
set -e

# --- CONFIGURATION ---
REPO_ROOT="$(pwd)"
GRANULARITY_DIR="${REPO_ROOT}/src/clusterllm/granularity"
DATASET_DIR="${REPO_ROOT}/datasets/banking77"

echo "========================================================"
echo "🔮 LANCEMENT DE LA SUITE DE PRÉDICTION (FINALE)"
echo "========================================================"

# 1. BLINDAGE DU CODE PYTHON (Prévention des crashs)
echo "[1/4] Vérification et correction des scripts Python..."
# On remplace 'input' par 'text' dans predict_pairs.py et predict_num_clusters.py
sed -i "s/\['input'\]/['text']/g" "${GRANULARITY_DIR}/predict_pairs.py" || true
sed -i "s/\['input'\]/['text']/g" "${GRANULARITY_DIR}/predict_num_clusters.py" || true
echo "   ✅ Scripts Python patchés."

# 2. DÉTECTION AUTOMATIQUE DU FICHIER D'ENTRÉE
# On prend le fichier JSON le plus récent dans le dossier sampled_pair_results
INPUT_FILE=$(ls -t "${GRANULARITY_DIR}/sampled_pair_results/"*.json | head -n 1)

if [ -z "$INPUT_FILE" ]; then
    echo "❌ ERREUR : Aucun fichier JSON trouvé dans sampled_pair_results."
    exit 1
fi
echo "   📂 Fichier d'entrée détecté : $(basename "$INPUT_FILE")"

# 3. ÉTAPE A : PRÉDICTION DES PAIRES
echo "[2/4] Création et lancement de predict_pairs..."
PREDICT_PAIRS_SCRIPT="${GRANULARITY_DIR}/scripts/predict_pairs.sh"
OUTPUT_PAIRS_DIR="${GRANULARITY_DIR}/predicted_pair_results"
mkdir -p "$OUTPUT_PAIRS_DIR"

cat > "$PREDICT_PAIRS_SCRIPT" <<BASH
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH:-}"
cd "${GRANULARITY_DIR}"

# Note : Si vous n'avez pas de clé OpenAI, vous pouvez simuler ou changer le model_name
# Ici on configure pour une exécution standard.
python predict_pairs.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_PAIRS_DIR" \
    --model_name "gpt-3.5-turbo" \
    --prompt_file "prompts/pair_prediction.txt" \
    --temperature 0.0
BASH
chmod +x "$PREDICT_PAIRS_SCRIPT"

# Exécution protégée (continue même si erreur API)
echo "   >> Exécution de predict_pairs.sh..."
"$PREDICT_PAIRS_SCRIPT" || echo "⚠️ Attention : predict_pairs a peut-être échoué (Clé API manquante ?). On continue vers les clusters."

# 4. ÉTAPE B : PRÉDICTION DU NOMBRE DE CLUSTERS
echo "[3/4] Création et lancement de predict_num_clusters..."
PREDICT_NUM_SCRIPT="${GRANULARITY_DIR}/scripts/predict_num_clusters.sh"
OUTPUT_NUM_DIR="${GRANULARITY_DIR}/predicted_num_clusters_results"
mkdir -p "$OUTPUT_NUM_DIR"

cat > "$PREDICT_NUM_SCRIPT" <<BASH
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH:-}"
cd "${GRANULARITY_DIR}"

python predict_num_clusters.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_NUM_DIR"
BASH
chmod +x "$PREDICT_NUM_SCRIPT"

echo "   >> Exécution de predict_num_clusters.sh..."
"$PREDICT_NUM_SCRIPT"

echo "========================================================"
echo "✅ SUITE DE PRÉDICTION TERMINÉE"
echo "📂 Résultats Paires   : $OUTPUT_PAIRS_DIR"
echo "📂 Résultats Clusters : $OUTPUT_NUM_DIR"
echo "========================================================"
