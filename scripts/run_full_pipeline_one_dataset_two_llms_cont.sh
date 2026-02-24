#!/usr/bin/env bash
# File name: scripts/run_full_pipeline_one_dataset_two_llms_cont.sh
#
# CONTRIBUTION PIPELINE (2 LLMs):
#   - llm_triplet: used ONLY for triplet prediction (Stage 3) -> drives finetune + FT embeddings
#   - llm_pairs  : used ONLY for pair prediction   (Stage 7.2) -> drives num_clusters estimation
#
# All CANONICAL artifacts are written with *_cont* names and stable paths.

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

ds="${ds:-massive_intent}"
sc="${sc:-small}"
llm_triplet="${llm_triplet:-deepseek-r1:32b}"
llm_pairs="${llm_pairs:-qwen2.5:32b}"

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

sanitize_tag() { echo "$1" | sed -E 's/[\/:[:space:]]+/__/g'; }
llm_triplet_dir="$(sanitize_tag "$llm_triplet")"
llm_pairs_dir="$(sanitize_tag "$llm_pairs")"
run_tag_cont="t=${llm_triplet_dir}__p=${llm_pairs_dir}__cont"

# -----------------------------
# Project directories
# -----------------------------
DATA_DIR="$ROOT/src/clusterllm/datasets"
PRED_TRIPLET_DIR="$ROOT/src/clusterllm/perspective/predict_triplet"
FT_DIR="$ROOT/src/clusterllm/perspective/finetuning"
GRAN_DIR="$ROOT/src/clusterllm/granularity"

# Canonical artifacts (ONLY these are meant to be tracked by DVC)
ART_DIR="$ROOT/runs_cont/artifacts_contrib_2llms_cont/$ds/$sc"
mkdir -p "$ART_DIR"

# Work dirs (can contain timestamped internals; not required for DVC)
WORK_DIR="$ROOT/runs_cont/_work_contrib_2llms/$ds/$sc/$run_tag_cont"
mkdir -p "$WORK_DIR"

# -----------------------------
# Helper: pick newest file from a glob
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

log "PIPELINE=CONTRIB_2LLMS_CONT"
log "ROOT=$ROOT"
log "dataset=$ds scale=$sc seed=$SEED"
log "llm_triplet=$llm_triplet llm_pairs=$llm_pairs"
log "run_tag_cont=$run_tag_cont"
log "OLLAMA_BASE_URL=$OLLAMA_BASE_URL"
log "TRIPLET_TEMPERATURE=$TRIPLET_TEMPERATURE PAIRS_TEMPERATURE=$PAIRS_TEMPERATURE"
log "ART_DIR=$ART_DIR"

# ------------------------------------------------------------------
# STAGE 0: data must already exist (CONT VERSION)
# ------------------------------------------------------------------
DATA_JSONL_CONT="$DATA_DIR/$ds/${sc}_cont.jsonl"
test -f "$DATA_JSONL_CONT" || die "Missing $DATA_JSONL_CONT. Run DVC stage: prepare_${sc}_cont (or prepare_small_cont/prepare_large_cont)."

# ------------------------------------------------------------------
# STAGE 1) Baseline embeddings -> *_embeds_contrib_cont.hdf5
# ------------------------------------------------------------------
log "Stage 1: baseline embeddings (CONTRIB)"
BASE_EMB_CONT="$DATA_DIR/$ds/${sc}_embeds_contrib_cont.hdf5"

python "$FT_DIR/get_embedding_cont.py" \
  --model_name "hkunlp/instructor-large" \
  --task_name "$ds" \
  --data_path "$DATA_JSONL_CONT" \
  --cache_dir "$ROOT/.cache/hf" \
  --result_file "$BASE_EMB_CONT" \
  --prompt "Represent the text for clustering." \
  --batch_size 512 \
  --scale "$sc" \
  --measure \
  --overwrite

test -f "$BASE_EMB_CONT" || die "Missing baseline embedding: $BASE_EMB_CONT"

# ------------------------------------------------------------------
# STAGE 2) Sample triplets
# ------------------------------------------------------------------
log "Stage 2: sample triplets"
TRIPLET_OUT="$WORK_DIR/triplets_cont"
mkdir -p "$TRIPLET_OUT"

python "$PRED_TRIPLET_DIR/triplet_sampling_cont.py" \
  --dataset "$ds" \
  --data_path "$DATA_JSONL_CONT" \
  --embed_path "$BASE_EMB_CONT" \
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
log "Stage 3: predict triplets with ollama (CONTRIB llm_triplet=$llm_triplet)"
TRIPLET_PRED_OUT="$WORK_DIR/triplet_preds_cont"
mkdir -p "$TRIPLET_PRED_OUT"

python "$PRED_TRIPLET_DIR/predict_cont.py" \
  --dataset "$ds" \
  --input_dir "$TRIPLET_OUT" \
  --output_dir "$TRIPLET_PRED_OUT" \
  --model_name "$llm_triplet" \
  --delay 0 \
  --max_trials 5 \
  --save_every 50 \
  --num_responses 1 \
  --num_threads 16 \
  --ollama-base-url "$OLLAMA_BASE_URL" \
  --ollama-model "$llm_triplet" \
  --ollama-timeout "$OLLAMA_TIMEOUT" \
  --ollama-temperature "$TRIPLET_TEMPERATURE" \
  --ollama-num-predict "$OLLAMA_NUM_PREDICT"

pred_json_raw="$(newest_file "$TRIPLET_PRED_OUT"/*.json)"
test -f "${pred_json_raw:-}" || die "No triplet prediction json found in $TRIPLET_PRED_OUT"

# CANONICAL (stable) file for DVC + downstream
PRED_TRIPLETS_CONT="$ART_DIR/pred_triplets_cont.json"
cp -f "$pred_json_raw" "$PRED_TRIPLETS_CONT"
test -f "$PRED_TRIPLETS_CONT" || die "Failed to write $PRED_TRIPLETS_CONT"
log "PRED_TRIPLETS_CONT=$PRED_TRIPLETS_CONT"

# ------------------------------------------------------------------
# STAGE 4) Convert triplets (standard + self optional) -> train_cont.json
# ------------------------------------------------------------------
log "Stage 4: convert triplets"
CONV_DIR="$WORK_DIR/converted_triplets_cont"
mkdir -p "$CONV_DIR"

(
  cd "$FT_DIR"
  python "convert_triplet_cont.py" \
    --dataset "$ds" \
    --pred_path "$PRED_TRIPLETS_CONT" \
    --output_path "$CONV_DIR" \
    --data_path "$DATA_JSONL_CONT"
)

standard_train_raw="$(newest_file "$CONV_DIR"/*train*.json)"
test -f "${standard_train_raw:-}" || die "convert_triplet_cont did not create a train json in $CONV_DIR"

(
  cd "$FT_DIR"
  python "convert_triplet_self_cont.py" \
    --dataset "$ds" \
    --pred_path "$PRED_TRIPLETS_CONT" \
    --feat_path "$BASE_EMB_CONT" \
    --output_path "$CONV_DIR" \
    --data_path "$DATA_JSONL_CONT" \
    || true
)

self_train_raw="$(newest_file "$CONV_DIR"/*self*train*.json)"
chosen_train_raw="$standard_train_raw"
if [[ -f "${self_train_raw:-}" ]]; then
  chosen_train_raw="$self_train_raw"
fi

TRAIN_CONT="$ART_DIR/train_cont.json"
cp -f "$chosen_train_raw" "$TRAIN_CONT"
test -f "$TRAIN_CONT" || die "Failed to write $TRAIN_CONT"
log "TRAIN_CONT=$TRAIN_CONT"

# ------------------------------------------------------------------
# STAGE 5) Finetune Instructor -> checkpoint dir (work)
# ------------------------------------------------------------------
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log "Stage 5: finetune instructor (CONTRIB)"
OUT_CKPT="$WORK_DIR/checkpoint_contrib_cont"
rm -rf "$OUT_CKPT"
mkdir -p "$OUT_CKPT"

python "$FT_DIR/finetune_cont.py" \
  --model_name_or_path "hkunlp/instructor-large" \
  --cache_dir "$ROOT/.cache/hf" \
  --train_file "$TRAIN_CONT" \
  --output_dir "$OUT_CKPT" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing True \
  --learning_rate 2e-6 \
  --num_train_epochs 1 \
  --bf16 \
  --logging_steps 5

# ------------------------------------------------------------------
# OPTIONAL PATCH (keeps pipeline portable)
# ------------------------------------------------------------------
log "Patch (idempotent): get_embedding safetensors support"
python "$ROOT/scripts/patch_get_embedding_safetensors.py" --repo-root "$ROOT" || true

# ------------------------------------------------------------------
# STAGE 6) Embeddings with finetuned checkpoint -> *_embeds_ft_contrib_cont.hdf5
# ------------------------------------------------------------------
log "Stage 6: finetuned embeddings (CONTRIB)"
FT_EMB_CONT="$DATA_DIR/$ds/${sc}_embeds_ft_contrib_cont.hdf5"

python "$FT_DIR/get_embedding_cont.py" \
  --model_name "hkunlp/instructor-large" \
  --task_name "$ds" \
  --data_path "$DATA_JSONL_CONT" \
  --cache_dir "$ROOT/.cache/hf" \
  --result_file "$FT_EMB_CONT" \
  --prompt "Represent the text for clustering." \
  --batch_size 512 \
  --checkpoint "$OUT_CKPT" \
  --scale "$sc" \
  --measure \
  --overwrite

test -f "$FT_EMB_CONT" || die "Missing finetuned embedding file: $FT_EMB_CONT"
log "FT_EMB_CONT=$FT_EMB_CONT"

# ------------------------------------------------------------------
# STAGE 7) Granularity
# ------------------------------------------------------------------

# 7.0 Sample pairs -> cluster_results_cont.json
log "Stage 7.0: sample pairs"
PAIR_OUT="$WORK_DIR/sampled_pairs_cont"
mkdir -p "$PAIR_OUT"

python "$GRAN_DIR/sample_pairs_cont.py" \
  --dataset "$ds" \
  --embed_method "instructor" \
  --data_path "$DATA_JSONL_CONT" \
  --feat_path "$FT_EMB_CONT" \
  --scale "$sc" \
  --k 1 \
  --out_dir "$PAIR_OUT" \
  --min_clusters 2 \
  --max_clusters 200 \
  --seed "$SEED"

cluster_raw="$(newest_file "$PAIR_OUT"/*.json)"
test -f "${cluster_raw:-}" || die "No cluster json in $PAIR_OUT"

CLUSTER_RESULTS_CONT="$ART_DIR/cluster_results_cont.json"
cp -f "$cluster_raw" "$CLUSTER_RESULTS_CONT"
test -f "$CLUSTER_RESULTS_CONT" || die "Failed to write $CLUSTER_RESULTS_CONT"
log "CLUSTER_RESULTS_CONT=$CLUSTER_RESULTS_CONT"

# 7.1 Prompt (run-specific) -> prompt_cont.json
log "Stage 7.1: generate prompt"
PROMPT_TEMPLATE="$FT_DIR/prompts.json"
test -f "$PROMPT_TEMPLATE" || die "Missing prompt template: $PROMPT_TEMPLATE"

PROMPT_RUN_DIR="$WORK_DIR/prompts_cont"
rm -rf "$PROMPT_RUN_DIR"
mkdir -p "$PROMPT_RUN_DIR"

python "$GRAN_DIR/sample_pairs_for_prompt_cont.py" \
  --prompt_path "$PROMPT_TEMPLATE" \
  --sampled_pair_path "$CLUSTER_RESULTS_CONT" \
  --data_path "$DATA_JSONL_CONT" \
  --dataset "$ds" \
  --num_sampled 200 \
  --num_for_prompt 4 \
  --seed "$SEED" \
  --out_dir "$PROMPT_RUN_DIR"

prompt_raw="$PROMPT_RUN_DIR/prompt.json"
test -f "$prompt_raw" || die "Prompt not created: $prompt_raw"

PROMPT_CONT="$ART_DIR/prompt_cont.json"
cp -f "$prompt_raw" "$PROMPT_CONT"
test -f "$PROMPT_CONT" || die "Failed to write $PROMPT_CONT"
log "PROMPT_CONT=$PROMPT_CONT"

# 7.2 Predict pairs (llm_pairs) -> pred_pairs_cont.json
log "Stage 7.2: predict pairs with ollama (CONTRIB llm_pairs=$llm_pairs)"
PPAIR_WORK="$WORK_DIR/_work_predict_pairs_cont"
rm -rf "$PPAIR_WORK"
mkdir -p "$PPAIR_WORK"
MARKER="$PPAIR_WORK/.start_predict_pairs_cont"
touch "$MARKER"

(
  cd "$PPAIR_WORK"
  python "$GRAN_DIR/predict_pairs_cont.py" \
    --dataset "$ds" \
    --data_path "$CLUSTER_RESULTS_CONT" \
    --prompt_file "$PROMPT_CONT" \
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

RAW_PRED_PAIRS="$(python - "$PPAIR_WORK" "$MARKER" <<'PY'
import os, sys
work = sys.argv[1]
marker = sys.argv[2]
t0 = os.path.getmtime(marker)
cands = []
for root, _, files in os.walk(work):
    for fn in files:
        if fn.endswith(".json"):
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

PRED_PAIRS_CONT="$ART_DIR/pred_pairs_cont.json"
cp -f "$RAW_PRED_PAIRS" "$PRED_PAIRS_CONT"
test -f "$PRED_PAIRS_CONT" || die "Failed to write $PRED_PAIRS_CONT"
log "PRED_PAIRS_CONT=$PRED_PAIRS_CONT"

# 7.3 predict_num_clusters -> num_clusters_cont.txt
log "Stage 7.3: predict num_clusters"
NUM_CLUSTERS_CONT="$ART_DIR/num_clusters_cont.txt"

python "$GRAN_DIR/predict_num_clusters_cont.py" \
  --dataset "$ds" \
  --data_path "$DATA_JSONL_CONT" \
  --clustering_results "$CLUSTER_RESULTS_CONT" \
  --pred_path "$PRED_PAIRS_CONT" \
  --scale "$sc" \
  --min_clusters 2 \
  --max_clusters 200 \
  | tee "$NUM_CLUSTERS_CONT"

test -f "$NUM_CLUSTERS_CONT" || die "Missing $NUM_CLUSTERS_CONT"

# ------------------------------------------------------------------
# FINAL) Write summary metric -> summary_cont.json
# ------------------------------------------------------------------
log "Write summary_cont.json (DVC metric)"
SUMMARY_DIR="$ROOT/runs_cont/summary/contrib_2llms_cont/$ds/$sc"
mkdir -p "$SUMMARY_DIR"
SUMMARY_CONT="$SUMMARY_DIR/summary_cont.json"

python - <<PY
import json, os

ds = ${ds@Q}
sc = ${sc@Q}
llm_triplet = ${llm_triplet@Q}
llm_pairs = ${llm_pairs@Q}
seed = int(${SEED@Q})

num_clusters_txt = ${NUM_CLUSTERS_CONT@Q}
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
  "pipeline": "contrib_2llms_cont",
  "dataset": ds,
  "scale": sc,
  "llm_triplet": llm_triplet,
  "llm_pairs": llm_pairs,
  "seed": seed,
  "ollama_base_url": ${OLLAMA_BASE_URL@Q},
  "triplet_temperature": float(${TRIPLET_TEMPERATURE@Q}),
  "pairs_temperature": float(${PAIRS_TEMPERATURE@Q}),
  "artifacts_dir": ${ART_DIR@Q},
  "pred_triplets_cont": ${PRED_TRIPLETS_CONT@Q},
  "train_cont": ${TRAIN_CONT@Q},
  "cluster_results_cont": ${CLUSTER_RESULTS_CONT@Q},
  "prompt_cont": ${PROMPT_CONT@Q},
  "pred_pairs_cont": ${PRED_PAIRS_CONT@Q},
  "num_clusters_cont": num_clusters_txt,
  "real_k": real_k,
  "estimated_k": estimated_k,
  "top10_candidates": top10,
}

os.makedirs(os.path.dirname(${SUMMARY_CONT@Q}), exist_ok=True)
with open(${SUMMARY_CONT@Q}, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("WROTE_SUMMARY_CONT:", ${SUMMARY_CONT@Q})
PY

log "DONE (CONTRIB_2LLMS_CONT). Summary: $SUMMARY_CONT"