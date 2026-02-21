#!/usr/bin/env bash
# File name: scripts/run_full_pipeline_one_dataset_two_llms.sh
# Usage:
#   chmod +x scripts/run_full_pipeline_one_dataset_two_llms.sh
#   ./scripts/run_full_pipeline_one_dataset_two_llms.sh
#
# Robust end-to-end pipeline (2 LLMs):
#   - llm_triplet: used ONLY for triplet prediction (Stage 3) -> drives finetune + FT embeddings
#   - llm_pairs  : used ONLY for pair prediction   (Stage 7.2) -> drives num_clusters estimation
#
# Stages:
#   stage0: data -> jsonl
#   stage1: baseline embeddings
#   stage2: sample triplets
#   stage3: predict triplets (ollama, llm_triplet)
#   stage4: convert triplets (standard + self)
#   stage5: finetune instructor (from triplets)
#   stage6: embeddings from finetuned checkpoint
#   stage7: granularity (sample_pairs -> prompt -> predict_pairs (llm_pairs) -> predict_num_clusters)

set -euo pipefail

# Auto-detect repo root = parent of scripts/
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

OLLAMA_BASE_URL="http://127.0.0.1:11434"
SEED=42

# ====== CHOOSE HERE ======
ds="massive_intent"          # any folder name inside: $ROOT/src/clusterllm/datasets/<ds>/
sc="small"                   # small | large

# 2 LLMs:
llm_triplet="deepseek-r1:32b"  # used at Stage 3 (triplets)
llm_pairs="qwen2.5:32b"        # used at Stage 7.2 (pairs)
# =========================

# ------------------------------------------------------------------
# Args override (no other edits needed)
# Usage:
#   ./scripts/run_full_pipeline_one_dataset_two_llms.sh <dataset> <scale> <llm_triplet> <llm_pairs> [seed]
# Example:
#   ./scripts/run_full_pipeline_one_dataset_two_llms.sh clinc150 small deepseek-r1:32b qwen2.5:32b 42
# ------------------------------------------------------------------

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 <dataset> <scale> <llm_triplet> <llm_pairs> [seed]" >&2
  exit 0
fi

if [[ -n "${1:-}" ]]; then ds="$1"; fi
if [[ -n "${2:-}" ]]; then sc="$2"; fi
if [[ -n "${3:-}" ]]; then llm_triplet="$3"; fi
if [[ -n "${4:-}" ]]; then llm_pairs="$4"; fi
if [[ -n "${5:-}" ]]; then SEED="$5"; fi

if [[ -z "${ds:-}" || -z "${sc:-}" || -z "${llm_triplet:-}" || -z "${llm_pairs:-}" ]]; then
  echo "ERROR: missing args. Need: <dataset> <scale> <llm_triplet> <llm_pairs> [seed]" >&2
  echo "Example: $0 clinc150 small deepseek-r1:32b qwen2.5:32b 42" >&2
  exit 1
fi

if [[ "$sc" != "small" && "$sc" != "large" ]]; then
  echo "ERROR: scale must be small|large, got: $sc" >&2
  exit 1
fi

llm_triplet_dir="$(echo "$llm_triplet" | sed 's/:/__/g')"
llm_pairs_dir="$(echo "$llm_pairs" | sed 's/:/__/g')"
run_tag="t=${llm_triplet_dir}__p=${llm_pairs_dir}"

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

DATA_DIR="$ROOT/src/clusterllm/datasets"
PREP="$ROOT/src/data/prepare_data.py"

# ------------------------------------------------------------------
# STAGE 0 REMOVED: data must already exist
# ------------------------------------------------------------------
test -f "$DATA_DIR/$ds/$sc.jsonl" || die "Missing $DATA_DIR/$ds/$sc.jsonl. Run: scripts/run_prepare_data_only.sh $ds $sc"

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
# STAGE 3) Predict triplets with Ollama LLM (llm_triplet)
# -> runs/perspective/triplet_preds/<ds>/<sc>/<run_tag>/<llm_triplet_dir>/
# ------------------------------------------------------------------
mkdir -p "$ROOT/runs/perspective/triplet_preds/$ds/$sc/$run_tag/$llm_triplet_dir"

python "$PRED_TRIPLET_DIR/predict.py" \
  --dataset "$ds" \
  --input_dir "$ROOT/runs/perspective/triplets/$ds/$sc" \
  --output_dir "$ROOT/runs/perspective/triplet_preds/$ds/$sc/$run_tag/$llm_triplet_dir" \
  --model_name "$llm_triplet" \
  --delay 0 \
  --max_trials 5 \
  --save_every 50 \
  --num_responses 1 \
  --ollama-base-url "$OLLAMA_BASE_URL" \
  --ollama-model "$llm_triplet" \
  --ollama-timeout 600 \
  --ollama-temperature 0.5 \
  --ollama-num-predict 10

PRED_DIR="$ROOT/runs/perspective/triplet_preds/$ds/$sc/$run_tag/$llm_triplet_dir"
pred_json="$(newest_file "$PRED_DIR"/*.json)"
test -f "$pred_json" || die "no pred json in $PRED_DIR"
echo "pred_json=$pred_json"

# ------------------------------------------------------------------
# STAGE 4) Convert triplets -> converted_triplets/<ds>/<sc>/<run_tag>/<llm_triplet_dir>/
# Run inside FT_DIR so prompts.json is found
# ------------------------------------------------------------------
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
# STAGE 5) Finetune Instructor (driven by llm_triplet triplets)
# ------------------------------------------------------------------
OUT_CKPT="$ROOT/runs/perspective/checkpoints/$ds/$sc/$run_tag/$llm_triplet_dir"
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

# 1) Ensure safetensors import
if "from safetensors.torch import load_file as safe_load_file" not in txt:
    m = re.search(r"(^\s*import\s+torch\s*$)", txt, flags=re.MULTILINE)
    if m:
        insert_at = m.end()
        txt = txt[:insert_at] + "\nfrom safetensors.torch import load_file as safe_load_file\n" + txt[insert_at:]
    else:
        # fallback: prepend (minimal)
        txt = "from safetensors.torch import load_file as safe_load_file\n" + txt

# 2) Prefer .bin, fallback to model.safetensors if missing (keep indentation)
pat_state_path = r'(^[ \t]*)state_path\s*=\s*os\.path\.join\(args\.checkpoint,\s*"pytorch_model\.bin"\)'
m = re.search(pat_state_path, txt, flags=re.MULTILINE)
if m and "model.safetensors" not in txt:
    indent = m.group(1)
    repl = (
        f'{indent}state_path = os.path.join(args.checkpoint, "pytorch_model.bin")\n'
        f'{indent}if not os.path.exists(state_path):\n'
        f'{indent}    alt = os.path.join(args.checkpoint, "model.safetensors")\n'
        f'{indent}    if os.path.exists(alt):\n'
        f'{indent}        state_path = alt'
    )
    txt = re.sub(pat_state_path, repl, txt, count=1, flags=re.MULTILINE)

# 3) Load safetensors when applicable (keep indentation)
pat_load = r'(^[ \t]*)state_dict\s*=\s*torch\.load\(state_path,\s*map_location="cpu"\)'
m = re.search(pat_load, txt, flags=re.MULTILINE)
if m:
    indent = m.group(1)
    repl = (
        f"{indent}if str(state_path).endswith('.safetensors'):\n"
        f"{indent}    state_dict = safe_load_file(str(state_path))\n"
        f"{indent}else:\n"
        f"{indent}    state_dict = torch.load(state_path, map_location=\"cpu\")"
    )
    txt = re.sub(pat_load, repl, txt, count=1, flags=re.MULTILINE)

path.write_text(txt, encoding="utf-8")
print("PATCHED:", path)
PY

# ------------------------------------------------------------------
# STAGE 6) Embeddings with finetuned checkpoint -> used in Granularity
# NOTE: FT embeddings depend on llm_triplet (because checkpoint does)
# ------------------------------------------------------------------
FT_EMB="$DATA_DIR/$ds/${sc}_embeds__ft__${run_tag}__${llm_triplet_dir}.hdf5"

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
# STAGE 7) Granularity: sample_pairs -> prompt -> predict_pairs (llm_pairs) -> predict_num_clusters
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

# 7.1 Generate prompt for THIS run
PROMPT_TEMPLATE="$FT_DIR/prompts.json"
test -f "$PROMPT_TEMPLATE" || die "missing prompt template: $PROMPT_TEMPLATE"

PROMPT_RUN_DIR="$ROOT/runs/granularity/prompts/$ds/$sc/$run_tag/$llm_pairs_dir"
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

# 7.2 predict_pairs (llm_pairs)
PPAIR_WORK="$ROOT/runs/granularity/_work_predict_pairs/$ds/$sc/$run_tag/$llm_pairs_dir"
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
    --ollama-model "$llm_pairs" \
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
        for k in ("results", "preds", "predictions", "pairs", "data"):
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

PPAIR_OUT="$ROOT/runs/granularity/predicted_pair_results/$ds/$sc/$run_tag/$llm_pairs_dir"
mkdir -p "$PPAIR_OUT"
pred_pairs_json="$PPAIR_OUT/pred_pairs.json"
cp -f "$RAW_PRED_PAIRS" "$pred_pairs_json"
echo "pred_pairs_json=$pred_pairs_json"

# 7.3 predict_num_clusters
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

echo "DONE. Result: $NC_OUT/num_clusters.txt"

# ------------------------------------------------------------------
# FINAL) Write a generic summary metric (stable path for DVC)
# ------------------------------------------------------------------
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
    txt = open(num_clusters_txt, "r", encoding="utf-8", errors="ignore").read().splitlines()
    for line in txt:
        if line.startswith("REAL K:"):
            real_k = int(line.split(":", 1)[1].strip())
        if line.startswith("ESTIMATED K:"):
            estimated_k = int(line.split(":", 1)[1].strip())
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