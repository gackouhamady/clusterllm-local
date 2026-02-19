#!/usr/bin/env bash
# File name: scripts/run_full_pipeline_one_dataset_one_llm.sh
# Usage:
#   chmod +x scripts/run_full_pipeline_one_dataset_one_llm.sh
#   ./scripts/run_full_pipeline_one_dataset_one_llm.sh
#
# Robust end-to-end pipeline:
#   stage0: data -> jsonl
#   stage1: baseline embeddings
#   stage2: sample triplets
#   stage3: predict triplets (ollama)
#   stage4: convert triplets (standard + self)
#   stage5: finetune instructor
#   stage6: embeddings from finetuned checkpoint
#   stage7: granularity (sample_pairs -> prompt -> predict_pairs -> predict_num_clusters)
#
# Assumption:
#   - finetune.py is the robust version (no strict assert on instruction; normalizes instead)

set -euo pipefail

# Auto-detect repo root = parent of scripts/
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

OLLAMA_BASE_URL="http://127.0.0.1:11434"
SEED=42

# ====== CHOOSE HERE ======
ds="massive_intent"                      # any folder name inside: $ROOT/src/clusterllm/datasets/<ds>/
sc="small"                       # small | large
llm="llama3.2:3b-instruct-q8_0"  # one model from `ollama list`
# =========================

llm_dir="$(echo "$llm" | sed 's/:/__/g')"

DATA_DIR="$ROOT/src/clusterllm/datasets"
PRED_TRIPLET_DIR="$ROOT/src/clusterllm/perspective/predict_triplet"
FT_DIR="$ROOT/src/clusterllm/perspective/finetuning"
GRAN_DIR="$ROOT/src/clusterllm/granularity"
PREP="$ROOT/src/data/prepare_data.py"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }

# Find newest file matching pattern; empty if none
newest_file() {
  local pattern="$1"
  ls -t $pattern 2>/dev/null | head -n 1 || true
}

# ------------------------------------------------------------------
# STAGE 0) Ensure data exists: $DATA_DIR/$ds/$sc.jsonl
# ------------------------------------------------------------------
mkdir -p "$DATA_DIR/$ds"

if [ ! -f "$DATA_DIR/$ds/$sc.jsonl" ]; then
  echo "MISSING: $DATA_DIR/$ds/$sc.jsonl -> running prepare_data.py to generate it"

  mkdir -p "$ROOT/data/raw"

  out_train_jsonl="$DATA_DIR/$ds/${sc}_train.jsonl"
  out_eval_jsonl="$DATA_DIR/$ds/${sc}_eval.jsonl"

  python "$PREP" \
    --dataset "$ds" \
    --seed "$SEED" \
    --output-dir "$ROOT/data/raw" \
    --datasets-dir "$DATA_DIR" \
    --split-train "$sc" \
    --split-eval "$sc" \
    --out-train "$ROOT/data/raw/${ds}_${sc}_train.csv" \
    --out-eval "$ROOT/data/raw/${ds}_${sc}_eval.csv" \
    --jsonl-train "$out_train_jsonl" \
    --jsonl-eval "$out_eval_jsonl"

  cat "$out_train_jsonl" "$out_eval_jsonl" > "$DATA_DIR/$ds/$sc.jsonl"
fi

test -f "$DATA_DIR/$ds/$sc.jsonl" || die "still missing $DATA_DIR/$ds/$sc.jsonl"

# ------------------------------------------------------------------
# STAGE 1) Embeddings (baseline Instructor) -> ${sc}_embeds.hdf5
# ------------------------------------------------------------------
python "$FT_DIR/get_embedding.py" \
  --model_name "hkunlp/instructor-large" \
  --task_name "$ds" \
  --data_path "$DATA_DIR/$ds/$sc.jsonl" \
  --cache_dir "$ROOT/.cache/hf" \
  --result_file "$DATA_DIR/$ds/${sc}_embeds.hdf5" \
  --prompt "Represent the text for clustering." \
  --batch_size 128 \
  --scale "$sc" \
  --measure \
  --overwrite

# ------------------------------------------------------------------
# STAGE 2) Sample triplets -> runs/perspective/triplets/<ds>/<sc>/
# ------------------------------------------------------------------
mkdir -p "$ROOT/runs/perspective/triplets/$ds/$sc"

python "$PRED_TRIPLET_DIR/triplet_sampling.py" \
  --dataset "$ds" \
  --data_path "$DATA_DIR/$ds/$sc.jsonl" \
  --embed_path "$DATA_DIR/$ds/${sc}_embeds.hdf5" \
  --output_dir "$ROOT/runs/perspective/triplets/$ds/$sc" \
  --scale "$sc" \
  --num_queries 1024 \
  --large_ent_prop 0.2 \
  --close_cluster_prop 0.0 \
  --max_distance 67 \
  --seed "$SEED"

# ------------------------------------------------------------------
# STAGE 3) Predict triplets with Ollama LLM -> runs/perspective/triplet_preds/<ds>/<sc>/<llm>/
# ------------------------------------------------------------------
mkdir -p "$ROOT/runs/perspective/triplet_preds/$ds/$sc/$llm_dir"

python "$PRED_TRIPLET_DIR/predict.py" \
  --dataset "$ds" \
  --input_dir "$ROOT/runs/perspective/triplets/$ds/$sc" \
  --output_dir "$ROOT/runs/perspective/triplet_preds/$ds/$sc/$llm_dir" \
  --model_name "$llm" \
  --delay 0 \
  --max_trials 5 \
  --save_every 50 \
  --num_responses 1 \
  --ollama-base-url "$OLLAMA_BASE_URL" \
  --ollama-model "$llm" \
  --ollama-timeout 600 \
  --ollama-temperature 0.5 \
  --ollama-num-predict 10

PRED_DIR="$ROOT/runs/perspective/triplet_preds/$ds/$sc/$llm_dir"
pred_json="$(newest_file "$PRED_DIR"/*.json)"
test -f "$pred_json" || die "no pred json in $PRED_DIR"
echo "pred_json=$pred_json"

# ------------------------------------------------------------------
# STAGE 4) Convert triplets -> converted_triplets/<ds>/<sc>/<llm>/
# Run inside FT_DIR so prompts.json is found
# ------------------------------------------------------------------
CONV_DIR="$ROOT/runs/perspective/converted_triplets/$ds/$sc/$llm_dir"
mkdir -p "$CONV_DIR"

(
  cd "$FT_DIR"
  python "convert_triplet.py" \
    --dataset "$ds" \
    --pred_path "$pred_json" \
    --output_path "$CONV_DIR" \
    --data_path "$DATA_DIR/$ds/$sc.jsonl"
)

standard_train_json="$(newest_file "$CONV_DIR"/*train*.json)"
test -f "$standard_train_json" || die "convert_triplet did not create a train json in $CONV_DIR"

(
  cd "$FT_DIR"
  python "convert_triplet_self.py" \
    --dataset "$ds" \
    --pred_path "$pred_json" \
    --feat_path "$DATA_DIR/$ds/${sc}_embeds.hdf5" \
    --output_path "$CONV_DIR" \
    --data_path "$DATA_DIR/$ds/$sc.jsonl" \
  || true
)

self_train_json="$(newest_file "$CONV_DIR"/*self*train*.json)"
if [ -f "${self_train_json:-}" ]; then
  train_json="$self_train_json"
else
  train_json="$standard_train_json"
fi
test -f "$train_json" || die "no train json found in $CONV_DIR"
echo "train_json=$train_json"

# ------------------------------------------------------------------
# STAGE 5) Finetune Instructor
# NOTE: assumes finetune.py is the robust version (instruction normalized, no strict assert)
# ------------------------------------------------------------------
OUT_CKPT="$ROOT/runs/perspective/checkpoints/$ds/$sc/$llm_dir"
rm -rf "$OUT_CKPT"
mkdir -p "$OUT_CKPT"

export PYTORCH_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "$FT_DIR/finetune.py" \
  --model_name_or_path "hkunlp/instructor-large" \
  --cache_dir "$ROOT/.cache/hf" \
  --train_file "$train_json" \
  --output_dir "$OUT_CKPT" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-6 \
  --num_train_epochs 1 \
  --bf16 \
  --logging_steps 5

# ------------------------------------------------------------------
# PATCH get_embedding.py to support safetensors checkpoints (idempotent)
# ------------------------------------------------------------------
python - <<'PY'
import pathlib, re

root = pathlib.Path.home() / "clusterllm-local"
path = root / "src/clusterllm/perspective/finetuning/get_embedding.py"
txt = path.read_text(encoding="utf-8")

if "from safetensors.torch import load_file as safe_load_file" not in txt:
    txt = re.sub(r"(import torch\s*\n)", r"\1from safetensors.torch import load_file as safe_load_file\n", txt, count=1)

if "state_dict = torch.load(state_path, map_location=\"cpu\")" in txt:
    txt = txt.replace(
        "state_dict = torch.load(state_path, map_location=\"cpu\")",
        "if str(state_path).endswith('.safetensors'):\n"
        "        state_dict = safe_load_file(str(state_path))\n"
        "    else:\n"
        "        state_dict = torch.load(state_path, map_location=\"cpu\")"
    )

if "state_path = os.path.join(args.checkpoint, \"pytorch_model.bin\")" in txt and "model.safetensors" not in txt:
    txt = txt.replace(
        "state_path = os.path.join(args.checkpoint, \"pytorch_model.bin\")",
        "state_path = os.path.join(args.checkpoint, \"pytorch_model.bin\")\n"
        "    if not os.path.exists(state_path):\n"
        "        alt = os.path.join(args.checkpoint, \"model.safetensors\")\n"
        "        if os.path.exists(alt):\n"
        "            state_path = alt\n"
    )

path.write_text(txt, encoding="utf-8")
print("PATCHED:", path)
PY

# ------------------------------------------------------------------
# STAGE 6) Embeddings with finetuned checkpoint -> used in Granularity
# ------------------------------------------------------------------
FT_EMB="$DATA_DIR/$ds/${sc}_embeds__ft__${llm_dir}.hdf5"

python "$FT_DIR/get_embedding.py" \
  --model_name "hkunlp/instructor-large" \
  --task_name "$ds" \
  --data_path "$DATA_DIR/$ds/$sc.jsonl" \
  --cache_dir "$ROOT/.cache/hf" \
  --result_file "$FT_EMB" \
  --prompt "Represent the text for clustering." \
  --batch_size 128 \
  --checkpoint "$OUT_CKPT" \
  --scale "$sc" \
  --measure \
  --overwrite

# ------------------------------------------------------------------
# STAGE 7) Granularity: sample_pairs -> prompt -> predict_pairs -> predict_num_clusters
# ------------------------------------------------------------------
feat_path="$FT_EMB"
test -f "$feat_path" || die "missing finetuned embedding file: $feat_path"
echo "feat_path=$feat_path"

# 7.0 Sample pairs
PAIR_OUT="$ROOT/runs/granularity/sampled_pairs/$ds/$sc"
mkdir -p "$PAIR_OUT"

python "$GRAN_DIR/sample_pairs.py" \
  --dataset "$ds" \
  --embed_method "instructor" \
  --data_path "$DATA_DIR/$ds/$sc.jsonl" \
  --feat_path "$feat_path" \
  --scale "$sc" \
  --k 1 \
  --out_dir "$PAIR_OUT" \
  --min_clusters 2 \
  --max_clusters 200 \
  --seed "$SEED"

cluster_json="$(newest_file "$PAIR_OUT"/*.json)"
test -f "${cluster_json:-}" || die "no cluster_json in $PAIR_OUT"
echo "cluster_json=$cluster_json"

# 7.1 Generate prompt for THIS run (copy into runs/)
PROMPT_TEMPLATE="$FT_DIR/prompts.json"
test -f "$PROMPT_TEMPLATE" || die "missing prompt template: $PROMPT_TEMPLATE"

PROMPT_RUN_DIR="$ROOT/runs/granularity/prompts/$ds/$sc/$llm_dir"
mkdir -p "$PROMPT_RUN_DIR"

python "$GRAN_DIR/sample_pairs_for_prompt.py" \
  --prompt_path "$PROMPT_TEMPLATE" \
  --sampled_pair_path "$cluster_json" \
  --data_path "$DATA_DIR/$ds/$sc.jsonl" \
  --dataset "$ds" \
  --num_sampled 200 \
  --num_for_prompt 4 \
  --seed "$SEED" \
|| true

GEN_PROMPT="$(newest_file "$GRAN_DIR/predicted_pair_results"/*.json)"
test -f "${GEN_PROMPT:-}" || die "sample_pairs_for_prompt produced no prompt json in $GRAN_DIR/predicted_pair_results"

PROMPT_RUN="$PROMPT_RUN_DIR/prompt.json"
cp -f "$GEN_PROMPT" "$PROMPT_RUN"
echo "PROMPT_RUN=$PROMPT_RUN"

# 7.2 predict_pairs (robust output detection)
PPAIR_WORK="$ROOT/runs/granularity/_work_predict_pairs/$ds/$sc/$llm_dir"
rm -rf "$PPAIR_WORK"
mkdir -p "$PPAIR_WORK"
echo "PPAIR_WORK=$PPAIR_WORK"

MARKER="$PPAIR_WORK/.start_predict_pairs"
touch "$MARKER"

(
  cd "$PPAIR_WORK"
  python "$GRAN_DIR/predict_pairs.py" \
    --dataset "$ds" \
    --data_path "$cluster_json" \
    --prompt_file "$PROMPT_TEMPLATE" \
    --delay 0 \
    --max_trials 5 \
    --save_every 50 \
    --overwrite \
    --ollama-base-url "$OLLAMA_BASE_URL" \
    --ollama-model "$llm" \
    --ollama-timeout 600 \
    --ollama-temperature 0.5 \
    --ollama-num-predict 10
)

CANDIDATES="$(
  find "$PPAIR_WORK" -type f -name "*.json" -newer "$MARKER" \
    ! -path "$PROMPT_RUN" \
    -print0 | xargs -0 -r ls -t 2>/dev/null || true
)"

RAW_PRED_PAIRS=""

for f in $CANDIDATES; do
  if python - "$f" <<'PY'
import json, sys
p = sys.argv[1]
try:
    x = json.load(open(p, "r", encoding="utf-8"))
except Exception:
    sys.exit(1)

def ok(obj):
    if isinstance(obj, list):
        return len(obj) > 0
    if isinstance(obj, dict):
        for k in ("results","preds","predictions","pairs","data"):
            if k in obj and isinstance(obj[k], list) and len(obj[k]) > 0:
                return True
        return len(obj) > 0
    return False

sys.exit(0 if ok(x) else 1)
PY
  then
    RAW_PRED_PAIRS="$f"
    break
  fi
done

test -f "${RAW_PRED_PAIRS:-}" || {
  echo "ERROR: could not identify predicted pairs json under $PPAIR_WORK"
  echo "DEBUG candidates (after marker):"
  printf "%s\n" $CANDIDATES | sed 's/^/  - /'
  echo "DEBUG all files:"
  find "$PPAIR_WORK" -maxdepth 6 -type f -print 2>/dev/null | sed 's/^/  - /'
  exit 1
}

echo "RAW_PRED_PAIRS=$RAW_PRED_PAIRS"

PPAIR_OUT="$ROOT/runs/granularity/predicted_pair_results/$ds/$sc/$llm_dir"
mkdir -p "$PPAIR_OUT"
pred_pairs_json="$PPAIR_OUT/pred_pairs.json"
cp -f "$RAW_PRED_PAIRS" "$pred_pairs_json"
echo "pred_pairs_json=$pred_pairs_json"

# 7.3 predict_num_clusters
NC_OUT="$ROOT/runs/granularity/num_clusters/$ds/$sc/$llm_dir"
mkdir -p "$NC_OUT"

python "$GRAN_DIR/predict_num_clusters.py" \
  --dataset "$ds" \
  --data_path "$DATA_DIR/$ds/$sc.jsonl" \
  --clustering_results "$cluster_json" \
  --pred_path "$pred_pairs_json" \
  --scale "$sc" \
  --min_clusters 2 \
  --max_clusters 200 \
  | tee "$NC_OUT/num_clusters.txt"

echo "DONE. Result: $NC_OUT/num_clusters.txt"
