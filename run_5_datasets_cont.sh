#!/bin/bash

echo "🚀 DÉMARRAGE DU BATCH : 5 DATASETS (CONTRIBUTIONS) - SMALL"

# 1. REDDIT
echo "▶️ 1/5 : Lancement de REDDIT..."
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=reddit" -S "run.scale=small" || true

# 2. CLINC150
echo "▶️ 2/5 : Lancement de CLINC150..."
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=clinc150" -S "run.scale=small" || true

# 3. MASSIVE INTENT
echo "▶️ 3/5 : Lancement de MASSIVE_INTENT..."
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=massive_intent" -S "run.scale=small" || true

# 4. ARXIV
echo "▶️ 4/5 : Lancement de ARXIV..."
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=arxiv" -S "run.scale=small" || true

# 5. FEW NERD NAT
echo "▶️ 5/5 : Lancement de FEW_NERD_NAT..."
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=few_nerd_nat" -S "run.scale=small" || true

echo "🎉 BATCH TERMINÉ !"
