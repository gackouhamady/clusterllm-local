#!/bin/bash

# 1. Définition des paramètres
DATASETS=("arxiv" "bank77" "clinc_domain" "clinc_intent" "few_event" "few_nerd_nat" "few_rel_nat" "go_emotions" "massive_domain" "massive_intent" "mtop_domain" "mtop_intent" "reddit" "stackex")
PIPELINES=("full_pipeline_2llms" "full_pipeline_2llms_cont")
SCALES=("small" "large")

LOG_FILE="rapport_execution_56.log"

echo "==========================================================" | tee -a $LOG_FILE
echo "🚀 DÉMARRAGE DU BATCH : 56 PIPELINES - $(date)" | tee -a $LOG_FILE
echo "==========================================================" | tee -a $LOG_FILE

# 2. Les 3 boucles d'exécution
for pipeline in "${PIPELINES[@]}"; do
    for scale in "${SCALES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            echo -e "\n⏳ [$(date +'%H:%M:%S')] DÉMARRAGE : $pipeline | $scale | dataset=$dataset" | tee -a $LOG_FILE
            
            # 3. Exécution avec capture
            if poetry run dvc exp run "$pipeline" -S "run.dataset=$dataset" -S "run.scale=$scale"; then
                echo "✅ [$(date +'%H:%M:%S')] SUCCÈS : $dataset ($scale, $pipeline) terminé proprement." | tee -a $LOG_FILE
            else
                echo "❌ [$(date +'%H:%M:%S')] ÉCHEC : $dataset ($scale, $pipeline) a planté. Passage au suivant..." | tee -a $LOG_FILE
            fi
            
        done
    done
done

echo -e "\n🎉 [$(date +'%H:%M:%S')] FIN TOTALE DES 56 PIPELINES." | tee -a $LOG_FILE
