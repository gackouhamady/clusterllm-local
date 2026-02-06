#!/usr/bin/env bash
set -e # Arrête tout dès la première erreur

# --- CONFIGURATION ---
REPO_ROOT="$(pwd)"
CHECKPOINT_DIR="${REPO_ROOT}/src/clusterllm/perspective/finetuning/checkpoints/banking77_finetuned"
CONFIG_FILE="${CHECKPOINT_DIR}/config.json"
EMBED_SCRIPT="${REPO_ROOT}/src/clusterllm/perspective/finetuning/scripts/get_embedding_finetuned.sh"
SAMPLE_SCRIPT="${REPO_ROOT}/src/clusterllm/granularity/scripts/sample_pairs.sh"
PYTHON_SAMPLE="${REPO_ROOT}/src/clusterllm/granularity/sample_pairs.py"
PKL_FILE="${REPO_ROOT}/datasets/banking77/embeddings_finetuned.pkl"

echo "========================================================"
echo "🚑 DÉMARRAGE DE LA PROCÉDURE DE SAUVETAGE"
echo "========================================================"

# 1. VERIFICATION ET REPARATION DU CHECKPOINT
echo "[1/4] Vérification du modèle finetuné..."
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERREUR CRITIQUE : Fichier config.json introuvable dans :"
    echo "$CHECKPOINT_DIR"
    echo "Le finetuning a-t-il vraiment fini ? Vérifiez le dossier checkpoints."
    exit 1
fi

# Injection forcée de l'identité MPNET via Python (plus sûr que sed)
python3 -c "
import json
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        data = json.load(f)
    
    if data.get('model_type') != 'mpnet':
        print('   🔧 Injection de model_type=\"mpnet\"...')
        data['model_type'] = 'mpnet'
        with open('$CONFIG_FILE', 'w') as f:
            json.dump(data, f, indent=2)
        print('   ✅ Config réparée.')
    else:
        print('   ✅ Config déjà correcte (mpnet).')
except Exception as e:
    print(f'   ❌ Erreur Python : {e}')
    sys.exit(1)
"

# 2. REPARATION DU CODE PYTHON (Colonne 'text' vs 'input')
echo "[2/4] Patch du code d'échantillonnage..."
# On remplace 'input' par 'text' si ce n'est pas déjà fait
sed -i "s/\['input'\]/['text']/g" "$PYTHON_SAMPLE"
echo "   ✅ Code Python aligné sur vos données."

# 3. GENERATION DES EMBEDDINGS
echo "[3/4] Génération du fichier .pkl (Embeddings)..."

# On recrée le script de lancement pour être sûr du chemin
cat > "$EMBED_SCRIPT" <<BASH
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="\$(cd "\$(dirname "\$0")/../../../../.." && pwd)"
export PYTHONPATH="\${REPO_ROOT}/src:\${PYTHONPATH:-}"
cd "\${REPO_ROOT}/src/clusterllm/perspective/finetuning"

python get_embedding.py \\
    --task_name "banking77" \\
    --data_path "\${REPO_ROOT}/datasets/banking77/test.jsonl" \\
    --result_file "\${REPO_ROOT}/datasets/banking77/embeddings_finetuned.pkl" \\
    --model_name "$CHECKPOINT_DIR" \\
    --batch_size 32 \\
    --overwrite
BASH
chmod +x "$EMBED_SCRIPT"

# Exécution
"$EMBED_SCRIPT"

if [ ! -f "$PKL_FILE" ]; then
    echo "❌ ERREUR : Le fichier .pkl n'a pas été créé !"
    exit 1
fi
echo "   ✅ Embeddings générés avec succès."

# 4. ECHANTILLONNAGE (GRANULARITÉ)
echo "[4/4] Lancement de l'échantillonnage des paires..."
chmod +x "$SAMPLE_SCRIPT"
"$SAMPLE_SCRIPT"

echo "========================================================"
echo "🎉 SUCCESS : TOUTES LES ÉTAPES SONT TERMINÉES !"
echo "========================================================"
