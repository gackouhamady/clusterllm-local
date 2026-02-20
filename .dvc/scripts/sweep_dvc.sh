# Persona-default run (recommended pair)
dvc exp run --queue \
  --set-param run.dataset=bank77 \
  --set-param run.fold=0 \
  --set-param run.persona=intent_discovery \
  --set-param run.triplet_llm="deepseek-r1:32b" \
  --set-param run.pairwise_llm="qwen2.5:32b" \
  --set-param run.run_dir="runs/bank77/deepseek-r1_32b__qwen2.5_32b/fold_0"

# Ablation: triplet model across ALL models (pairwise fixed)
for m in $(yq -r '.ollama.models[]' params.yaml); do
  safe=$(echo "$m" | sed 's/[:.]/_/g')
  dvc exp run --queue \
    --set-param run.dataset=bank77 \
    --set-param run.fold=0 \
    --set-param run.persona=intent_discovery \
    --set-param run.triplet_llm="$m" \
    --set-param run.pairwise_llm="qwen2.5:32b" \
    --set-param run.run_dir="runs/bank77/${safe}__qwen2_5_32b/fold_0"
done

# Then execute queued experiments:
dvc exp run --run-all
dvc exp show
