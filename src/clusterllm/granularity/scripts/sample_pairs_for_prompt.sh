#!/usr/bin/env bash
set -euo pipefail

# 1. Définition manuelle de la racine pour éviter toute erreur
REPO_ROOT="/home/hamadygackou777/clusterllm-local"
GRAN_DIR="${REPO_ROOT}/src/clusterllm/granularity"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 2. Se placer dans le dossier où se trouve sample_pairs_for_prompt.py
if [ -d "$GRAN_DIR" ]; then
    cd "$GRAN_DIR"
    echo "📍 Dossier de travail : $PWD"
else
    echo "❌ ERREUR : Le dossier $GRAN_DIR n'existe pas."
    exit 1
fi

# 3. Variables
prompt_path="prompts_pair_exps_pair_v8.json"

for dataset in "banking77"
do
    echo "--- Traitement de : $dataset ---"
    
    # Chemins basés sur la racine REPO_ROOT
    sampled_pair_path="${GRAN_DIR}/sampled_pair_results/${dataset}_embed=finetuned_s=small_k=1_multigran2-200_seed=100.json"
    data_path="${REPO_ROOT}/datasets/${dataset}/test.jsonl"

    # Vérification de l'existence du fichier de données
    if [ ! -f "$data_path" ]; then
        echo "❌ ERREUR : Fichier de données introuvable à : $data_path"
        continue
    fi

    python sample_pairs_for_prompt.py \
        --prompt_path "$prompt_path" \
        --sampled_pair_path "$sampled_pair_path" \
        --data_path "$data_path" \
        --dataset "$dataset" \
        --num_sampled 16 \
        --num_for_prompt 2 \
        --seed 1234
done

echo "✅ Script terminé."