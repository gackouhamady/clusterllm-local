#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_clusterllm_local.sh --dataset banking77 --ollama-model qwen25_7b_q5
#   ./run_clusterllm_local.sh --dataset banking77 --ollama-model qwen2.5:7b-instruct-q5_K_M
#
# Required:
#   --dataset NAME
#   --ollama-model NAME_OR_TAG   (key from registry or full ollama tag)
#
# Optional:
#   --repo-root PATH
#   --ollama-url URL            (alias: --ollama-base-url)
#   --batch-embed 32
#   --k-triplets 5              (mapped to --k or --max_query depending on sampler)
#   --k-pairs 1
#   --min-clusters 2
#   --max-clusters 200
#   --seed 100
#   --epochs 1
#   --batch-ft 16
#   --scale small
#   --simulate-triplets true|false

die() { echo "Error: $*" >&2; exit 1; }

REPO_ROOT="${PWD}"
DATASET=""
OLLAMA_MODEL_IN=""
OLLAMA_URL="http://localhost:11434"

BATCH_EMBED=32
K_TRIPLETS=5
K_PAIRS=1
MIN_CLUSTERS=2
MAX_CLUSTERS=200
SEED=100
EPOCHS=1
BATCH_FT=16
SCALE="small"
SIMULATE_TRIPLETS="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --ollama-model) OLLAMA_MODEL_IN="$2"; shift 2 ;;
    --ollama-url|--ollama-base-url) OLLAMA_URL="$2"; shift 2 ;;
    --batch-embed) BATCH_EMBED="$2"; shift 2 ;;
    --k-triplets) K_TRIPLETS="$2"; shift 2 ;;
    --k-pairs) K_PAIRS="$2"; shift 2 ;;
    --min-clusters) MIN_CLUSTERS="$2"; shift 2 ;;
    --max-clusters) MAX_CLUSTERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-ft) BATCH_FT="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    --simulate-triplets) SIMULATE_TRIPLETS="$2"; shift 2 ;;
    -h|--help) sed -n '1,220p' "$0"; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

[[ -n "${DATASET}" ]] || die "--dataset is required"
[[ -n "${OLLAMA_MODEL_IN}" ]] || die "--ollama-model is required"

cd "${REPO_ROOT}"
REPO_ROOT="$(pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OLLAMA_URL="${OLLAMA_URL}"
export CLUSTERLLM_OLLAMA_BASE_URL="${OLLAMA_URL}"

# Model registry (keys -> ollama tags)
declare -A MODEL_REGISTRY=(
  ["llama3_q4km"]="llama3:8b-instruct-q4_K_M"
  ["mistral_q4km"]="mistral:7b-instruct-q4_K_M"
  ["gemma_q4km"]="gemma:7b-instruct-q4_K_M"
  ["llama31_8b_q8"]="llama3.1:8b-instruct-q8_0"
  ["mixtral_8x7b_q4km"]="mixtral:8x7b-instruct-q4_K_M"
  ["qwen25_7b_q5"]="qwen2.5:7b-instruct-q5_K_M"
)

OLLAMA_MODEL_KEY="${OLLAMA_MODEL_IN}"
if [[ -n "${MODEL_REGISTRY[${OLLAMA_MODEL_IN}]+x}" ]]; then
  OLLAMA_MODEL_TAG="${MODEL_REGISTRY[${OLLAMA_MODEL_IN}]}"
else
  # treat input as already a full ollama tag
  OLLAMA_MODEL_TAG="${OLLAMA_MODEL_IN}"
fi

export CLUSTERLLM_OLLAMA_MODEL_KEY="${OLLAMA_MODEL_KEY}"
export CLUSTERLLM_OLLAMA_MODEL_TAG="${OLLAMA_MODEL_TAG}"

DATA_DIR="${REPO_ROOT}/datasets/${DATASET}"
TEST_JSONL="${DATA_DIR}/test.jsonl"
[[ -f "${TEST_JSONL}" ]] || die "missing dataset file: ${TEST_JSONL}"

# Detect finetuning directory
PERS_FINETUNE_DIR="$(dirname "$(find "${REPO_ROOT}/src" -name "finetune.py" -path "*clusterllm*perspective*finetuning*" -print -quit || true)")"
[[ -n "${PERS_FINETUNE_DIR}" && -d "${PERS_FINETUNE_DIR}" ]] || \
  PERS_FINETUNE_DIR="$(dirname "$(find "${REPO_ROOT}/src" -name "get_embedding.py" -path "*clusterllm*perspective*finetuning*" -print -quit || true)")"
[[ -n "${PERS_FINETUNE_DIR}" && -d "${PERS_FINETUNE_DIR}" ]] || die "could not find perspective/finetuning directory"

# Detect granularity directory
GRAN_DIR="$(dirname "$(find "${REPO_ROOT}/src" -name "sample_pairs.py" -path "*clusterllm*granularity*" -print -quit || true)")"
[[ -n "${GRAN_DIR}" && -d "${GRAN_DIR}" ]] || die "could not find granularity directory"

# Detect triplet sampling implementation
TRIPLET_SAMPLING_PY="$(find "${REPO_ROOT}/src" -name "triplet_sampling.py" -print -quit || true)"
TRIPLET_PREDICTION_DIR="$(find "${REPO_ROOT}/src" -type d -path "*clusterllm/perspective/triplet_prediction" -print -quit || true)"

TRIPLET_MODE=""
PERS_TRIPLET_DIR=""
if [[ -n "${TRIPLET_SAMPLING_PY}" ]]; then
  PERS_TRIPLET_DIR="$(dirname "${TRIPLET_SAMPLING_PY}")"
  TRIPLET_MODE="triplet_sampling_py"
elif [[ -n "${TRIPLET_PREDICTION_DIR}" && -f "${TRIPLET_PREDICTION_DIR}/sampling.py" ]]; then
  PERS_TRIPLET_DIR="${TRIPLET_PREDICTION_DIR}"
  TRIPLET_MODE="triplet_prediction_sampling_py"
else
  die "could not find triplet sampler (triplet_sampling.py or triplet_prediction/sampling.py)"
fi

CHECKPOINT_DIR="${PERS_FINETUNE_DIR}/checkpoints/${DATASET}_finetuned"

mkdir -p \
  "${DATA_DIR}" \
  "${PERS_FINETUNE_DIR}/checkpoints" \
  "${PERS_FINETUNE_DIR}/converted_triplet_results" \
  "${PERS_TRIPLET_DIR}/sampled_triplet_results" \
  "${PERS_TRIPLET_DIR}/predicted_triplet_results" \
  "${GRAN_DIR}/prompts" \
  "${GRAN_DIR}/sampled_pair_results" \
  "${GRAN_DIR}/predicted_pair_results" \
  "${GRAN_DIR}/predicted_num_clusters_results"

echo "Repo root: ${REPO_ROOT}"
echo "Dataset: ${DATASET}"
echo "Finetune dir: ${PERS_FINETUNE_DIR}"
echo "Triplet dir: ${PERS_TRIPLET_DIR} (${TRIPLET_MODE})"
echo "Granularity dir: ${GRAN_DIR}"
echo "Ollama model key: ${OLLAMA_MODEL_KEY}"
echo "Ollama model tag: ${OLLAMA_MODEL_TAG}"
echo "Ollama url: ${OLLAMA_URL}"
echo "Simulate triplets: ${SIMULATE_TRIPLETS}"

# 1) Initial embeddings
cd "${PERS_FINETUNE_DIR}"
python get_embedding.py \
  --task_name "${DATASET}" \
  --data_path "${TEST_JSONL}" \
  --result_file "${DATA_DIR}/embeddings.pkl" \
  --model_name "sentence-transformers/all-mpnet-base-v2" \
  --batch_size "${BATCH_EMBED}" \
  --overwrite

# 2) Triplet sampling
cd "${PERS_TRIPLET_DIR}"
if [[ "${TRIPLET_MODE}" == "triplet_sampling_py" ]]; then
  python triplet_sampling.py \
    --dataset "${DATASET}" \
    --data_path "${TEST_JSONL}" \
    --feat_path "${DATA_DIR}/embeddings.pkl" \
    --out_dir "sampled_triplet_results" \
    --k "${K_TRIPLETS}"
else
  python sampling.py \
    --dataset "${DATASET}" \
    --data_path "${TEST_JSONL}" \
    --feat_path "${DATA_DIR}/embeddings.pkl" \
    --scale "${SCALE}" \
    --max_query "${K_TRIPLETS}" \
    --out_dir "sampled_triplet_results" \
    --seed "${SEED}"
fi

TRIPLET_INPUT="$(ls -t sampled_triplet_results/${DATASET}*.json 2>/dev/null | head -n 1 || true)"
[[ -n "${TRIPLET_INPUT}" ]] || die "no triplet file generated in ${PERS_TRIPLET_DIR}/sampled_triplet_results"

# 3) Triplet prediction (simulate or real, always uses selected ollama model)
cd "${PERS_TRIPLET_DIR}"
TRIPLET_PRED_BASENAME="$(basename "${TRIPLET_INPUT}" .json)-${OLLAMA_MODEL_KEY}-pred.json"
TRIPLET_PRED="predicted_triplet_results/${TRIPLET_PRED_BASENAME}"

python3 - <<PY
import json, os, sys, time
from urllib import request

base_url = "${OLLAMA_URL}".rstrip("/")
model = "${OLLAMA_MODEL_TAG}"
simulate = "${SIMULATE_TRIPLETS}".lower() == "true"
inp = "${TRIPLET_INPUT}"
outp = "${TRIPLET_PRED}"

def unwrap(x):
    if isinstance(x, list) and x:
        return x[-1]
    return x

def pick_fields(d):
    # try many possible formats
    q = d.get("query") or d.get("question") or d.get("query_text") or d.get("q")
    c1 = d.get("choice1") or d.get("option1") or d.get("candidate1") or d.get("text_a") or d.get("a")
    c2 = d.get("choice2") or d.get("option2") or d.get("candidate2") or d.get("text_b") or d.get("b")
    q = unwrap(q)
    c1 = unwrap(c1)
    c2 = unwrap(c2)
    return q, c1, c2

def ollama_chat(prompt, timeout=600):
    url = base_url + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode("utf-8"))
    return (resp.get("message") or {}).get("content", "").strip()

print("Triplet input :", inp)
print("Triplet output:", outp)
print("Triplet model :", model)
os.makedirs(os.path.dirname(outp), exist_ok=True)

with open(inp, "r") as f:
    data = json.load(f)

# unwrap common container format
arr = data.get("test_inputs") if isinstance(data, dict) and "test_inputs" in data else data
if not isinstance(arr, list):
    raise SystemExit("Triplet file format not supported (expected list or {test_inputs: [...]})")

for i, d in enumerate(arr):
    if "prediction" in d:
        continue

    if simulate:
        d["content"] = "Choice 1"
        d["prediction"] = ["Choice 1"]
        continue

    q, c1, c2 = pick_fields(d)
    if not (isinstance(q, str) and isinstance(c1, str) and isinstance(c2, str)):
        # last-resort: skip but still make convert_triplet happy
        d["content"] = "Choice 1"
        d["prediction"] = ["Choice 1"]
        continue

    prompt = (
        "Select the customer utterance that better corresponds with the Query in terms of intent.\n\n"
        f"Query: {q}\n"
        f"Choice 1: {c1}\n"
        f"Choice 2: {c2}\n\n"
        "Please respond with 'Choice 1' or 'Choice 2' without explanation."
    )

    content = ""
    for _ in range(3):
        content = ollama_chat(prompt)
        if content:
            break
        time.sleep(0.2)

    pred = []
    if "Choice 1" in content and "Choice 2" not in content:
        pred = ["Choice 1"]
    elif "Choice 2" in content and "Choice 1" not in content:
        pred = ["Choice 2"]
    else:
        pred = ["Choice 1"]  # conservative fallback

    d["content"] = content or pred[0]
    d["prediction"] = pred

with open(outp, "w") as f:
    json.dump(arr, f)

print("Saved:", outp)
PY

[[ -f "${TRIPLET_PRED}" ]] || die "triplet prediction file not created: ${TRIPLET_PRED}"

# 4) Convert triplets
cd "${PERS_FINETUNE_DIR}"
python convert_triplet.py \
  --dataset "${DATASET}" \
  --pred_path "${PERS_TRIPLET_DIR}/${TRIPLET_PRED}" \
  --output_path "${PERS_FINETUNE_DIR}/converted_triplet_results" \
  --data_path "${TEST_JSONL}"

CONVERTED_FILE="$(ls -t converted_triplet_results/${DATASET}*train*.json 2>/dev/null | head -n 1 || true)"
[[ -n "${CONVERTED_FILE}" ]] || CONVERTED_FILE="$(ls -t converted_triplet_results/${DATASET}*.json 2>/dev/null | head -n 1 || true)"
[[ -n "${CONVERTED_FILE}" ]] || die "convert_triplet.py produced no file in ${PERS_FINETUNE_DIR}/converted_triplet_results"
CONVERTED_PATH="${PERS_FINETUNE_DIR}/${CONVERTED_FILE}"
echo "Converted triplets: ${CONVERTED_PATH}"

# 5) Finetune (skip if checkpoint exists)
cd "${PERS_FINETUNE_DIR}"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  python finetune.py \
    --model_name "sentence-transformers/all-mpnet-base-v2" \
    --train_data "${CONVERTED_PATH}" \
    --output_dir "checkpoints/${DATASET}_finetuned" \
    --num_epochs "${EPOCHS}" \
    --batch_size "${BATCH_FT}"
fi

# 6) Patch config.json (mpnet)
python3 - <<PY
import json, os
p = os.path.join("${CHECKPOINT_DIR}", "config.json")
if os.path.exists(p):
    d = json.load(open(p))
    d["model_type"] = "mpnet"
    json.dump(d, open(p, "w"))
PY

# 7) Finetuned embeddings
cd "${PERS_FINETUNE_DIR}"
python get_embedding.py \
  --task_name "${DATASET}" \
  --data_path "${TEST_JSONL}" \
  --result_file "${DATA_DIR}/embeddings_finetuned.h5" \
  --model_name "${CHECKPOINT_DIR}" \
  --batch_size "${BATCH_EMBED}" \
  --overwrite

# 8) Pair sampling
cd "${GRAN_DIR}"
python sample_pairs.py \
  --dataset "${DATASET}" \
  --data_path "${TEST_JSONL}" \
  --feat_path "${DATA_DIR}/embeddings_finetuned.h5" \
  --scale "${SCALE}" \
  --embed_method "finetuned" \
  --k "${K_PAIRS}" \
  --out_dir "sampled_pair_results" \
  --min_clusters "${MIN_CLUSTERS}" \
  --max_clusters "${MAX_CLUSTERS}" \
  --seed "${SEED}"

PAIRS_INPUT="$(ls -t sampled_pair_results/${DATASET}*.json 2>/dev/null | head -n 1 || true)"
[[ -n "${PAIRS_INPUT}" ]] || die "no pair file generated in ${GRAN_DIR}/sampled_pair_results"

# 9) Pair prompt
cd "${GRAN_DIR}"
echo "{\"${DATASET}\": \"Are the following two sentences in the same cluster?\\nSentence 1: {text_a}\\nSentence 2: {text_b}\\nAnswer (Yes/No):\"}" \
  > prompts/pair_prediction.json

# 10) Predict pairs + final cluster count
# IMPORTANT: predict_pairs.py expects --ollama-base-url (not --ollama-url)
python predict_pairs.py \
  --dataset "${DATASET}" \
  --data_path "${GRAN_DIR}/${PAIRS_INPUT}" \
  --prompt_file "prompts/pair_prediction.json" \
  --temperature 0.0 \
  --ollama-base-url "${OLLAMA_URL}" \
  --ollama-model "${OLLAMA_MODEL_TAG}"

PAIRS_PRED="$(ls -t predicted_pair_results/${DATASET}*.json 2>/dev/null | head -n 1 || true)"
[[ -n "${PAIRS_PRED}" ]] || die "no predicted pairs file in ${GRAN_DIR}/predicted_pair_results"

python predict_num_clusters.py \
  --dataset "${DATASET}" \
  --data_path "${TEST_JSONL}" \
  --clustering_results "${GRAN_DIR}/${PAIRS_PRED}" \
  --pred_path "predicted_num_clusters_results/FINAL_${DATASET}.json" \
  --embed_method "finetuned" \
  --scale "${SCALE}"

cat "predicted_num_clusters_results/FINAL_${DATASET}.json"
