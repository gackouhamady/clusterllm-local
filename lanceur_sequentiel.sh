#!/bin/bash

echo "🚀 DÉMARRAGE DU SCRIPT SÉQUENTIEL DES 56 PIPELINES"

# Sécurité absolue : on casse tout verrou DVC fantôme avant de commencer
pkill -9 -f dvc || true
rm -f /home/hamady_gackou_work/clusterllm-local/.dvc/tmp/rwlock

# =========================================================
# PHASE 1 : full_pipeline_2llms (SMALL)
# =========================================================
echo "▶️ Phase 1/4 : full_pipeline_2llms - SMALL"
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=arxiv" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=bank77" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=clinc_domain" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=clinc_intent" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=few_event" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=few_nerd_nat" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=few_rel_nat" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=go_emotions" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=massive_domain" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=massive_intent" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=mtop_domain" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=mtop_intent" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=reddit" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=stackex" -S "run.scale=small" || true

# =========================================================
# PHASE 2 : full_pipeline_2llms (LARGE)
# =========================================================
echo "▶️ Phase 2/4 : full_pipeline_2llms - LARGE"
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=arxiv" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=bank77" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=clinc_domain" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=clinc_intent" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=few_event" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=few_nerd_nat" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=few_rel_nat" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=go_emotions" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=massive_domain" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=massive_intent" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=mtop_domain" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=mtop_intent" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=reddit" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms -S "run.dataset=stackex" -S "run.scale=large" || true

# =========================================================
# PHASE 3 : full_pipeline_2llms_cont (SMALL)
# =========================================================
echo "▶️ Phase 3/4 : full_pipeline_2llms_cont - SMALL"
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=arxiv" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=bank77" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=clinc_domain" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=clinc_intent" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=few_event" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=few_nerd_nat" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=few_rel_nat" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=go_emotions" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=massive_domain" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=massive_intent" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=mtop_domain" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=mtop_intent" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=reddit" -S "run.scale=small" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=stackex" -S "run.scale=small" || true

# =========================================================
# PHASE 4 : full_pipeline_2llms_cont (LARGE)
# =========================================================
echo "▶️ Phase 4/4 : full_pipeline_2llms_cont - LARGE"
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=arxiv" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=bank77" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=clinc_domain" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=clinc_intent" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=few_event" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=few_nerd_nat" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=few_rel_nat" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=go_emotions" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=massive_domain" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=massive_intent" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=mtop_domain" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=mtop_intent" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=reddit" -S "run.scale=large" || true
poetry run dvc exp run full_pipeline_2llms_cont -S "run.dataset=stackex" -S "run.scale=large" || true

echo "🎉 FIN TOTALE DES 56 COMMANDES !"
