#!/usr/bin/env bash
# File name: scripts/run_full_pipeline_one_dataset_two_llms.sh
#
# Robust end-to-end pipeline (2 LLMs):
#   - llm_triplet: used ONLY for triplet prediction (Stage 3) -> drives finetune + FT embeddings
#   - llm_pairs  : used ONLY for pair prediction   (Stage 7.2) -> drives num_clusters estimation
#
# Stages:
#   stage0: data -> jsonl (handled by separate DVC stages / scripts)
#   stage1: baseline embeddings
#   stage2: sample triplets
#   stage3: predict triplets (ollama, llm_triplet)
#   stage4: convert triplets (standard + self)
#   stage5: finetune instructor (from triplets)
#   stage6: embeddings from finetuned checkpoint
#   stage7: granularity (sample_pairs -> prompt -> predict_pairs (llm_pairs) -> predict_num_clusters)

set -euo pipefail

# -----------------------------
# Repo root and PYTHONPATH
# -----------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

# -----------------------------
# Defaults (override via args)
# -----------------------------
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
SEED="${SEED:-42}"

# choose defaults here
ds="${ds:-massive_intent}"          # dataset folder name in: $ROOT/src/clusterllm/datasets/<ds>/
sc="${sc:-small}"                   # small | large
llm_triplet="${llm_triplet:-deepseek-r1:32b}"  # Stage 3
llm_pairs="${llm_pairs:-qwen2.5:32b}"          # Stage 7.2

# LLM decoding (more reproducible when temp low)
TRIPLET_TEMPERATURE="${TRIPLET_TEMPERATURE:-0.2}"
PAIRS_TEMPERATURE="${PAIRS_TEMPERATURE:-0.0}"
OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-600}"
OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT:-16}"

# -----------------------------
# Usage
# -----------------------------
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat >&2 <<EOF
Usage: $0 <dataset> <scale> <llm_triplet> <llm_pairs> [seed]

Example:
  $0 clinc150 small deepseek-r1:32b qwen2.5:32b 42

Environment overrides (optional):
  OLLAMA_BASE_URL, TRIPLET_TEMPERATURE, PAIRS_TEMPERATURE, OLLAMA_TIMEOUT, OLLAMA_NUM_PREDICT
EOF
  exit 0
fi

if [[ -n "${1:-}" ]]; then ds="$1"; fi
if [[ -n "${2:-}" ]]; then sc="$2"; fi
if [[ -n "${3:-}" ]]; then llm_triplet="$3"; fi
if [[ -n "${4:-}" ]]; then llm_pairs="$4"; fi
if [[ -n "${5:-}" ]]; then SEED="$5"; fi

# -----------------------------
# Validation
# -----------------------------
die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

[[ -n "${ds:-}" && -n "${sc:-}" && -n "${llm_triplet:-}" && -n "${llm_pairs:-}" ]] \
  || die "Missing args. Need: <dataset> <scale> <llm_triplet> <llm_pairs> [seed]"

[[ "$sc" == "small" || "$sc" == "large" ]] || die "scale must be small|large, got: $sc"

command -v python >/dev/null 2>&1 || die "python not found in PATH"
command -v bash >/dev/null 2>&1 || die "bash not found in PATH"

# -----------------------------
# Path-safe tags
# -----------------------------
sanitize_tag() {
  # replace / : space with __ to make safe folder names
  echo "$1" | sed -E 's/[\/:[:space:]]+/__/g'
}

llm_triplet_dir="$(sanitize_tag "$llm_triplet")"
llm_pairs_dir="$(sanitize_tag "$llm_pairs")"
run_tag="t=${llm_triplet_dir}__p=${llm_pairs_dir}"

# -----------------------------
# Project directories
# -----------------------------
DATA_DIR="$ROOT/src/clusterllm/datasets"
PRED_TRIPLET_DIR="$ROOT/src/clusterllm/perspective/predict_triplet"
FT_DIR="$ROOT/src/clusterllm/perspective/finetuning"
GRAN_DIR="$ROOT/src/clusterllm/granularity"

# -----------------------------
# Helper: pick newest file from a glob (robust)
# -----------------------------
newest_file() {
  local pattern="$1"
  python - "$pattern" <<'PY'
import glob, os, sys
pat = sys.argv[1]
files = glob.glob(pat)
if not files:
    sys.exit(0)
latest = max(files, key=lambda p: os.path.getmtime(p))
print(latest)
PY
}

log "ROOT=$ROOT"
log "dataset=$ds scale=$sc seed=$SEED"
log "llm_triplet=$llm_triplet llm_pairs=$llm_pairs"
log "run_tag=$run_tag"
log "OLLAMA_BASE_URL=$OLLAMA_BASE_URL"
log "TRIPLET_TEMPERATURE=$TRIPLET_TEMPERATURE PAIRS_TEMPERATURE=$PAIRS_TEMPERATURE"

# ------------------------------------------------------------------
# STAGE 0: data must already exist
# ------------------------------------------------------------------
test -f "$DATA_DIR/$ds/$sc.jsonl" || die "Missing $DATA_DIR/$ds/$sc.jsonl. Run: scripts/run_prepare_data_only.sh $ds $sc $SEED"

# ------------------------------------------------------------------
# STAGE 1) Baseline embeddings -> ${sc}_embeds.hdf5
# ------------------------------------------------------------------
log "Stage 1: baseline embeddings"
python "$FT_DIR/get_embedding.py" \
  --model_name "hkunlp/instructor-large" \
  --task_name "$ds" \
  --data_path "$DATA_DIR/$ds/$sc.jsonl" \
  --cache_dir "$ROOT/.cache/hf" \
  --result_file "$DATA_DIR/$ds/${sc}_embeds.hdf5" \
  --prompt "Represent the text for clustering." \
  --batch_size 512 \
  --scale "$sc" \
  --measure \
  --overwrite

# ------------------------------------------------------------------
# STAGE 2) Sample triplets
# ------------------------------------------------------------------
log "Stage 2: sample triplets"
TRIPLET_OUT="$ROOT/runs/perspective/triplets/$ds/$sc"
mkdir -p "$TRIPLET_OUT"

python "$PRED_TRIPLET_DIR/triplet_sampling.py" \
  --dataset "$ds" \
  --data_path "$DATA_DIR/$ds/$sc.jsonl" \
  --embed_path "$DATA_DIR/$ds/${sc}_embeds.hdf5" \
  --output_dir "$TRIPLET_OUT" \
  --scale "$sc" \
  --num_queries 1024 \
  --large_ent_prop 0.2 \
  --close_cluster_prop 0.0 \
  --max_distance 67 \
  --seed "$SEED"

# ------------------------------------------------------------------
# STAGE 3) Predict triplets (llm_triplet)
# ------------------------------------------------------------------
log "Stage 3: predict triplets with ollama ($llm_triplet)"
TRIPLET_PRED_OUT="$ROOT/runs/perspective/triplet_preds/$ds/$sc/$run_tag/$llm_triplet_dir"
mkdir -p "$TRIPLET_PRED_OUT"

python "$PRED_TRIPLET_DIR/predict.py" \
  --dataset "$ds" \
  --input_dir "$TRIPLET_OUT" \
  --output_dir "$TRIPLET_PRED_OUT" \
  --model_name "$llm_triplet" \
  --delay 0 \
  --max_trials 5 \
  --save_every 50 \
  --num_responses 1 \
  --ollama-base-url "$OLLAMA_BASE_URL" \
  --ollama-model "$llm_triplet" \
  --ollama-timeout "$OLLAMA_TIMEOUT" \
  --ollama-temperature "$TRIPLET_TEMPERATURE" \
  --ollama-num-predict "$OLLAMA_NUM_PREDICT"

pred_json="$(newest_file "$TRIPLET_PRED_OUT"/*.json)"
test -f "${pred_json:-}" || die "No triplet prediction json found in $TRIPLET_PRED_OUT"
log "pred_json=$pred_json"

# ------------------------------------------------------------------
# STAGE 4) Convert triplets
# ------------------------------------------------------------------
log "Stage 4: convert triplets"
CONV_DIR="$ROOT/runs/perspective/converted_triplets/$ds/$sc/$run_tag/$llm_triplet_dir"
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
test -f "${standard_train_json:-}" || die "convert_triplet did not create a train json in $CONV_DIR"

# try self conversion (optional)
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
if [[ -f "${self_train_json:-}" ]]; then
  train_json="$self_train_json"
else
  train_json="$standard_train_json"
fi
test -f "${train_json:-}" || die "No train json found in $CONV_DIR"
log "train_json=$train_json"

# ------------------------------------------------------------------
# STAGE 5) Finetune Instructor
# ------------------------------------------------------------------
log "Stage 5: finetune instructor"
OUT_CKPT="$ROOT/runs/perspective/checkpoints/$ds/$sc/$run_tag/$llm_triplet_dir"
rm -rf "$OUT_CKPT"
mkdir -p "$OUT_CKPT"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "$FT_DIR/finetune.py" \
  --model_name_or_path "hkunlp/instructor-large" \
  --cache_dir "$ROOT/.cache/hf" \
  --train_file "$train_json" \
  --output_dir "$OUT_CKPT" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-6 \
  --num_train_epochs 1 \
  --bf16 \
  --logging_steps 5

# ------------------------------------------------------------------
# OPTIONAL PATCH: safetensors support for get_embedding.py (portable)
# (You should ideally fix get_embedding.py directly, but this keeps pipeline stable.)
# ------------------------------------------------------------------
log "Patch (idempotent): get_embedding.py safetensors support"
python "$ROOT/scripts/patch_get_embedding_safetensors.py" --repo-root "$ROOT" || true

# ------------------------------------------------------------------
# STAGE 6) Embeddings with finetuned checkpoint
# ------------------------------------------------------------------
log "Stage 6: finetuned embeddings"
FT_EMB="$DATA_DIR/$ds/${sc}_embeds__ft__${run_tag}__${llm_triplet_dir}.hdf5"

python "$FT_DIR/get_embedding.py" \
  --model_name "hkunlp/instructor-large" \
  --task_name "$ds" \
  --data_path "$DATA_DIR/$ds/$sc.jsonl" \
  --cache_dir "$ROOT/.cache/hf" \
  --result_file "$FT_EMB" \
  --prompt "Represent the text for clustering." \
  --batch_size 512 \
  --checkpoint "$OUT_CKPT" \
  --scale "$sc" \
  --measure \
  --overwrite

test -f "$FT_EMB" || die "Missing finetuned embedding file: $FT_EMB"
log "feat_path=$FT_EMB"

# ------------------------------------------------------------------
# STAGE 7) Granularity
# ------------------------------------------------------------------

# 7.0 Sample pairs
log "Stage 7.0: sample pairs"
PAIR_OUT="$ROOT/runs/granularity/sampled_pairs/$ds/$sc/$run_tag"
mkdir -p "$PAIR_OUT"

python "$GRAN_DIR/sample_pairs.py" \
  --dataset "$ds" \
  --embed_method "instructor" \
  --data_path "$DATA_DIR/$ds/$sc.jsonl" \
  --feat_path "$FT_EMB" \
  --scale "$sc" \
  --k 1 \
  --out_dir "$PAIR_OUT" \
  --min_clusters 2 \
  --max_clusters 200 \
  --seed "$SEED"

cluster_json="$(newest_file "$PAIR_OUT"/*.json)"
test -f "${cluster_json:-}" || die "No cluster_json in $PAIR_OUT"
log "cluster_json=$cluster_json"

# 7.1 Generate prompt for THIS run (DO NOT overwrite template)
log "Stage 7.1: generate prompt"
PROMPT_TEMPLATE="$FT_DIR/prompts.json"
test -f "$PROMPT_TEMPLATE" || die "Missing prompt template: $PROMPT_TEMPLATE"

PROMPT_RUN_DIR="$ROOT/runs/granularity/prompts/$ds/$sc/$run_tag/$llm_pairs_dir"
rm -rf "$PROMPT_RUN_DIR"
mkdir -p "$PROMPT_RUN_DIR"
PROMPT_RUN="$PROMPT_RUN_DIR/prompt.json"

python "$GRAN_DIR/sample_pairs_for_prompt.py" \
  --prompt_path "$PROMPT_TEMPLATE" \
  --sampled_pair_path "$cluster_json" \
  --data_path "$DATA_DIR/$ds/$sc.jsonl" \
  --dataset "$ds" \
  --num_sampled 200 \
  --num_for_prompt 4 \
  --seed "$SEED" \
  --out_dir "$PROMPT_RUN_DIR"

test -f "$PROMPT_RUN" || die "Prompt not created: $PROMPT_RUN"
log "PROMPT_RUN=$PROMPT_RUN"

# 7.2 predict_pairs (IMPORTANT: use PROMPT_RUN, not PROMPT_TEMPLATE)
log "Stage 7.2: predict pairs with ollama ($llm_pairs)"
PPAIR_WORK="$ROOT/runs/granularity/_work_predict_pairs/$ds/$sc/$run_tag/$llm_pairs_dir"
rm -rf "$PPAIR_WORK"
mkdir -p "$PPAIR_WORK"

MARKER="$PPAIR_WORK/.start_predict_pairs"
touch "$MARKER"

(
  cd "$PPAIR_WORK"
  python "$GRAN_DIR/predict_pairs.py" \
    --dataset "$ds" \
    --data_path "$cluster_json" \
    --prompt_file "$PROMPT_RUN" \
    --delay 0 \
    --max_trials 5 \
    --save_every 50 \
    --overwrite \
    --ollama-base-url "$OLLAMA_BASE_URL" \
    --ollama-model "$llm_pairs" \
    --ollama-timeout "$OLLAMA_TIMEOUT" \
    --ollama-temperature "$PAIRS_TEMPERATURE" \
    --ollama-num-predict "$OLLAMA_NUM_PREDICT"
)

# find newest json after marker (predictions)
RAW_PRED_PAIRS="$(python - "$PPAIR_WORK" "$MARKER" <<'PY'
import os, sys
work = sys.argv[1]
marker = sys.argv[2]
t0 = os.path.getmtime(marker)
cands = []
for root, _, files in os.walk(work):
    for fn in files:
        if not fn.endswith(".json"):
            continue
        p = os.path.join(root, fn)
        if os.path.getmtime(p) > t0:
            cands.append(p)
if not cands:
    sys.exit(0)
cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
print(cands[0])
PY
)"

test -f "${RAW_PRED_PAIRS:-}" || die "Could not find predicted pairs json under $PPAIR_WORK"
log "RAW_PRED_PAIRS=$RAW_PRED_PAIRS"

PPAIR_OUT="$ROOT/runs/granularity/predicted_pair_results/$ds/$sc/$run_tag/$llm_pairs_dir"
mkdir -p "$PPAIR_OUT"
pred_pairs_json="$PPAIR_OUT/pred_pairs.json"
cp -f "$RAW_PRED_PAIRS" "$pred_pairs_json"
log "pred_pairs_json=$pred_pairs_json"

# 7.3 predict_num_clusters
log "Stage 7.3: predict num_clusters"
NC_OUT="$ROOT/runs/granularity/num_clusters/$ds/$sc/$run_tag/$llm_pairs_dir"
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

log "DONE. Result: $NC_OUT/num_clusters.txt"

# ------------------------------------------------------------------
# FINAL) Write summary metric (stable path for DVC)
# ------------------------------------------------------------------
log "Write summary.json (DVC metric)"
SUMMARY_DIR="$ROOT/runs/summary/full_pipeline_2llms/$ds/$sc"
mkdir -p "$SUMMARY_DIR"
SUMMARY_JSON="$SUMMARY_DIR/summary.json"

python - <<PY
import json, os

ds = ${ds@Q}
sc = ${sc@Q}
llm_triplet = ${llm_triplet@Q}
llm_pairs = ${llm_pairs@Q}
seed = int(${SEED@Q})
run_tag = ${run_tag@Q}

num_clusters_txt = ${NC_OUT@Q} + "/num_clusters.txt"
estimated_k = None
top10 = None
real_k = None

if os.path.exists(num_clusters_txt):
    lines = open(num_clusters_txt, "r", encoding="utf-8", errors="ignore").read().splitlines()
    for line in lines:
        if line.startswith("REAL K:"):
            try: real_k = int(line.split(":", 1)[1].strip())
            except: pass
        if line.startswith("ESTIMATED K:"):
            try: estimated_k = int(line.split(":", 1)[1].strip())
            except: pass
        if line.startswith("TOP 10 CANDIDATES:"):
            rhs = line.split(":", 1)[1].strip()
            try:
                top10 = json.loads(rhs.replace("'", '"'))
            except Exception:
                top10 = rhs

out = {
  "dataset": ds,
  "scale": sc,
  "llm_triplet": llm_triplet,
  "llm_pairs": llm_pairs,
  "seed": seed,
  "run_tag": run_tag,
  "ollama_base_url": ${OLLAMA_BASE_URL@Q},
  "triplet_temperature": float(${TRIPLET_TEMPERATURE@Q}),
  "pairs_temperature": float(${PAIRS_TEMPERATURE@Q}),
  "num_clusters_txt": num_clusters_txt,
  "real_k": real_k,
  "estimated_k": estimated_k,
  "top10_candidates": top10,
}

os.makedirs(os.path.dirname(${SUMMARY_JSON@Q}), exist_ok=True)
with open(${SUMMARY_JSON@Q}, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("WROTE_SUMMARY:", ${SUMMARY_JSON@Q})
PY