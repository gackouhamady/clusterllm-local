#!/usr/bin/env bash
set -euo pipefail

# 1. Optimisation Mémoire
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Chemins du projet
REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 3. Dossier de travail
WORK_DIR="${REPO_ROOT}/src/clusterllm/perspective/finetuning"
echo "Déplacement vers : $WORK_DIR"
cd "$WORK_DIR"

# 4. Variables & Gestion du modèle source
# ==> LE PARAMÈTRE $1 DÉFINIT LE LLM QUI A CRÉÉ LES DONNÉES (ex: qwen2.5:7b)
LLM_SOURCE="${1:-llama3_q4km}"

# Le modèle qu'on va entraîner (ne change pas, sauf si vous voulez changer d'embedding)
MODEL_TO_FINETUNE="hkunlp/instructor-large"

# Construction dynamique du nom de fichier d'entrée
TRAIN_FILE="${WORK_DIR}/converted_triplet_results/banking77_embed=instructor_s=small_m=500_d=67_choice_seed=42-${LLM_SOURCE}-train.json"

# Dossier de sortie
OUTPUT_DIR="${WORK_DIR}/checkpoints/banking77_finetuned_instructor_${LLM_SOURCE}"

mkdir -p "$OUTPUT_DIR"

# Vérification de sécurité
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ ERREUR : Le fichier d'entraînement est introuvable !"
    echo "   Chemin cherché : $TRAIN_FILE"
    echo "   Astuce : Avez-vous converti les données venant du modèle '$LLM_SOURCE' ?"
    exit 1
fi

echo "----------------------------------------------------------------"
echo "Démarrage du Finetuning"
echo "Données source (LLM) : $LLM_SOURCE"
echo "Modèle à entraîner   : $MODEL_TO_FINETUNE"
echo "Fichier entrée       : $TRAIN_FILE"
echo "Sortie               : $OUTPUT_DIR"
echo "----------------------------------------------------------------"

# 5. Exécution
python finetune.py \
    --train_file "$TRAIN_FILE" \
    --model_name "$MODEL_TO_FINETUNE" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-5 \
    --cl_temperature 0.05