#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_clusterllm_local.sh --dataset banking77 --ollama-model mistral_q4km
#   ./run_clusterllm_local.sh --dataset banking77 --ollama-model mistral:7b-instruct-q4_K_M
#
# Required:
#   --dataset NAME
#   --ollama-model NAME_OR_TAG
#
# Optional:
#   --repo-root PATH
#   --ollama-url URL                 (base url, default http://localhost:11434)
#   --batch-embed 32
#   --k-triplets 5                   (mapped to --k or --max_query depending on sampler)
#   --k-pairs 1
#   --min-clusters 2
#   --max-clusters 200
#   --seed 100
#   --epochs 1
#   --batch-ft 16
#   --scale small
#   --simulate-triplets true|false   (if true, always picks Choice 1)

# Model registry (keys -> Ollama tags)
declare -A MODEL_REGISTRY=(
  ["llama3_q4km"]="llama3:8b-instruct-q4_K_M"
  ["mistral_q4km"]="mistral:7b-instruct-q4_K_M"
  ["gemma_q4km"]="gemma:7b-instruct-q4_K_M"
  ["llama31_8b_q8"]="llama3.1:8b-instruct-q8_0"
  ["mixtral_8x7b_q4km"]="mixtral:8x7b-instruct-q4_K_M"
  ["qwen25_7b_q5"]="qwen2.5:7b-instruct-q5_K_M"
)

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
SIMULATE_TRIPLETS="false"

show_help() {
  sed -n '1,120p' "$0"
  echo
  echo "Known --ollama-model keys:"
  for k in "${!MODEL_REGISTRY[@]}"; do
    printf "  %-18s -> %s\n" "$k" "${MODEL_REGISTRY[$k]}"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --ollama-model) OLLAMA_MODEL_IN="$2"; shift 2 ;;
    --ollama-url) OLLAMA_URL="$2"; shift 2 ;;
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
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${DATASET}" ]]; then
  echo "Error: --dataset is required" >&2
  exit 2
fi
if [[ -z "${OLLAMA_MODEL_IN}" ]]; then
  echo "Error: --ollama-model is required" >&2
  exit 2
fi

cd "${REPO_ROOT}"
REPO_ROOT="$(pwd)"

# Resolve model key/tag
OLLAMA_MODEL_KEY="${OLLAMA_MODEL_IN}"
OLLAMA_MODEL_TAG=""
if [[ -n "${MODEL_REGISTRY[$OLLAMA_MODEL_IN]+x}" ]]; then
  OLLAMA_MODEL_TAG="${MODEL_REGISTRY[$OLLAMA_MODEL_IN]}"
else
  # treat input as direct Ollama tag
  OLLAMA_MODEL_TAG="${OLLAMA_MODEL_IN}"
fi

# safe file suffix
SAFE_MODEL_SUFFIX="$(echo "${OLLAMA_MODEL_KEY}" | tr '/:' '__')"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATA_DIR="${REPO_ROOT}/datasets/${DATASET}"
TEST_JSONL="${DATA_DIR}/test.jsonl"
if [[ ! -f "${TEST_JSONL}" ]]; then
  echo "Error: missing dataset file: ${TEST_JSONL}" >&2
  exit 1
fi

# Detect directories
PERS_FINETUNE_DIR="$(dirname "$(find "${REPO_ROOT}/src" -name "get_embedding.py" -path "*clusterllm*perspective*finetuning*" -print -quit || true)")"
GRAN_DIR="$(dirname "$(find "${REPO_ROOT}/src" -name "sample_pairs.py" -path "*clusterllm*granularity*" -print -quit || true)")"

if [[ -z "${PERS_FINETUNE_DIR}" || ! -d "${PERS_FINETUNE_DIR}" ]]; then
  echo "Error: could not find perspective/finetuning directory" >&2
  exit 1
fi
if [[ -z "${GRAN_DIR}" || ! -d "${GRAN_DIR}" ]]; then
  echo "Error: could not find granularity directory" >&2
  exit 1
fi

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
  echo "Error: could not find triplet sampler (triplet_sampling.py or triplet_prediction/sampling.py)" >&2
  exit 1
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
if [[ -z "${TRIPLET_INPUT}" ]]; then
  echo "Error: no triplet file generated in ${PERS_TRIPLET_DIR}/sampled_triplet_results" >&2
  exit 1
fi

# 3) Triplet prediction + conversion (robust; replaces convert_triplet.py)
cd "${PERS_TRIPLET_DIR}"
TRIPLET_PRED="predicted_triplet_results/$(basename "${TRIPLET_INPUT}" .json)-${SAFE_MODEL_SUFFIX}-pred.json"
CONVERTED_PATH="${PERS_FINETUNE_DIR}/converted_triplet_results/${DATASET}-${SAFE_MODEL_SUFFIX}-train.json"

python3 - <<PY
import json, os, re, sys
from urllib import request, error

dataset = "${DATASET}"
test_jsonl = "${TEST_JSONL}"
sampled_path = os.path.join("${PERS_TRIPLET_DIR}", "${TRIPLET_INPUT}")
pred_path = os.path.join("${PERS_TRIPLET_DIR}", "${TRIPLET_PRED}")
out_path = "${CONVERTED_PATH}"

ollama_url = "${OLLAMA_URL}".rstrip("/")
model_tag = "${OLLAMA_MODEL_TAG}"
simulate = "${SIMULATE_TRIPLETS}".lower() == "true"

def load_jsonl_texts(path):
    texts = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if isinstance(obj, str):
                texts.append(obj)
                continue
            if isinstance(obj, dict):
                for k in ("text","sentence","utterance","query","input"):
                    v = obj.get(k)
                    if isinstance(v, str):
                        texts.append(v)
                        break
                else:
                    # fallback: first string field
                    s = None
                    for v in obj.values():
                        if isinstance(v, str):
                            s = v; break
                    texts.append(s if s is not None else "")
            else:
                texts.append("")
    return texts

def ollama_call(prompt):
    # try /api/chat then fallback to /api/generate
    chat_url = ollama_url + "/api/chat"
    gen_url = ollama_url + "/api/generate"
    payload_chat = {"model": model_tag, "messages": [{"role":"user","content": prompt}], "stream": False}
    data = json.dumps(payload_chat).encode("utf-8")
    req = request.Request(chat_url, data=data, headers={"Content-Type":"application/json"})
    try:
        with request.urlopen(req, timeout=120) as resp:
            obj = json.load(resp)
            msg = obj.get("message", {})
            return (msg.get("content") or "").strip()
    except Exception:
        payload_gen = {"model": model_tag, "prompt": prompt, "stream": False}
        data2 = json.dumps(payload_gen).encode("utf-8")
        req2 = request.Request(gen_url, data=data2, headers={"Content-Type":"application/json"})
        with request.urlopen(req2, timeout=120) as resp:
            obj = json.load(resp)
            return (obj.get("response") or "").strip()

def parse_choice(content):
    c = (content or "").strip()
    c_low = c.lower()
    # accept "Choice 1/2"
    if ("choice 1" in c_low) and ("choice 2" not in c_low):
        return 1
    if ("choice 2" in c_low) and ("choice 1" not in c_low):
        return 2
    # accept bare 1/2
    has1 = re.search(r"\b1\b", c_low) is not None
    has2 = re.search(r"\b2\b", c_low) is not None
    if has1 and not has2:
        return 1
    if has2 and not has1:
        return 2
    return None

def extract_from_input(inp):
    # Expected format:
    # Query: ...
    # Choice 1: ...
    # Choice 2: ...
    m = re.search(r"Query:\s*(.*?)\nChoice 1:\s*(.*?)\nChoice 2:\s*(.*?)\nChoice", inp, flags=re.S)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

texts = load_jsonl_texts(test_jsonl)

with open(sampled_path, "r") as f:
    sampled = json.load(f)
if isinstance(sampled, dict) and "test_inputs" in sampled:
    sampled = sampled["test_inputs"]
if not isinstance(sampled, list):
    raise RuntimeError("Unexpected sampled triplet format (expected list or dict with test_inputs).")

pred_items = []
out_data = []

for item in sampled:
    q = c1 = c2 = None

    # Preferred: indices
    qi = item.get("query_idx")
    c1i = item.get("choice1_idx")
    c2i = item.get("choice2_idx")
    if isinstance(qi, int) and isinstance(c1i, int) and isinstance(c2i, int) and qi < len(texts) and c1i < len(texts) and c2i < len(texts):
        q, c1, c2 = texts[qi], texts[c1i], texts[c2i]

    # Fallback: parse from "input"
    if (q is None or c1 is None or c2 is None) and isinstance(item.get("input"), str):
        parsed = extract_from_input(item["input"])
        if parsed:
            q, c1, c2 = parsed

    if not q or not c1 or not c2:
        continue

    prompt = (
        "Select the banking customer utterance that better corresponds with the Query in terms of intent.\n\n"
        f"Query: {q}\n"
        f"Choice 1: {c1}\n"
        f"Choice 2: {c2}\n\n"
        "Please respond with 'Choice 1' or 'Choice 2' without explanation."
    )

    if simulate:
        choice = 1
        content = "Choice 1"
    else:
        content = ollama_call(prompt)
        choice = parse_choice(content) or 1

    pred = ["Choice 1"] if choice == 1 else ["Choice 2"]
    pos = c1 if choice == 1 else c2
    neg = c2 if choice == 1 else c1

    pred_items.append({
        "task_name": dataset,
        "query_idx": qi,
        "choice1_idx": c1i,
        "choice2_idx": c2i,
        "content": content,
        "prediction": pred,
        "query": q,
        "pos": pos,
        "neg": neg,
    })
    out_data.append({"query": q, "pos": pos, "neg": neg})

# Save predicted triplets (debuggable)
os.makedirs(os.path.dirname(pred_path), exist_ok=True)
with open(pred_path, "w") as f:
    json.dump(pred_items, f, indent=2)

# Save converted triplets for finetune
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out_data, f, indent=2)

print("Triplet input :", sampled_path)
print("Triplet output:", pred_path)
print("Triplet model :", model_tag)
print("Converted triplets:", out_path)
print("Converted count:", len(out_data))
if len(out_data) == 0:
    raise SystemExit("Error: no converted triplets produced (out_data is empty).")
PY

# 4) Finetune (skip if checkpoint exists)
cd "${PERS_FINETUNE_DIR}"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  python finetune.py \
    --model_name "sentence-transformers/all-mpnet-base-v2" \
    --train_data "${CONVERTED_PATH}" \
    --output_dir "checkpoints/${DATASET}_finetuned" \
    --num_epochs "${EPOCHS}" \
    --batch_size "${BATCH_FT}"
fi

# 5) Patch config.json (mpnet)
python3 - <<PY
import json, os
p = os.path.join("${CHECKPOINT_DIR}", "config.json")
if os.path.exists(p):
    d = json.load(open(p))
    d["model_type"] = "mpnet"
    json.dump(d, open(p, "w"))
PY

# 6) Finetuned embeddings
cd "${PERS_FINETUNE_DIR}"
python get_embedding.py \
  --task_name "${DATASET}" \
  --data_path "${TEST_JSONL}" \
  --result_file "${DATA_DIR}/embeddings_finetuned.h5" \
  --model_name "${CHECKPOINT_DIR}" \
  --batch_size "${BATCH_EMBED}" \
  --overwrite

# 7) Pair sampling
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
if [[ -z "${PAIRS_INPUT}" ]]; then
  echo "Error: no pair file generated in ${GRAN_DIR}/sampled_pair_results" >&2
  exit 1
fi

# 8) Pair prompt
echo "{\"${DATASET}\": \"Are the following two sentences in the same cluster?\\nSentence 1: {text_a}\\nSentence 2: {text_b}\\nAnswer (Yes/No):\"}" \
  > prompts/pair_prediction.json

# 9) Predict pairs (note: predict_pairs.py expects --ollama-base-url, not --ollama-url)
python predict_pairs.py \
  --dataset "${DATASET}" \
  --data_path "${GRAN_DIR}/${PAIRS_INPUT}" \
  --prompt_file "prompts/pair_prediction.json" \
  --temperature 0.0 \
  --ollama-model "${OLLAMA_MODEL_TAG}" \
  --ollama-base-url "${OLLAMA_URL}"

PAIRS_PRED="$(ls -t predicted_pair_results/${DATASET}*.json 2>/dev/null | head -n 1 || true)"
if [[ -z "${PAIRS_PRED}" ]]; then
  echo "Error: no predicted pairs file in ${GRAN_DIR}/predicted_pair_results" >&2
  exit 1
fi

python predict_num_clusters.py \
  --dataset "${DATASET}" \
  --data_path "${TEST_JSONL}" \
  --clustering_results "${GRAN_DIR}/${PAIRS_PRED}" \
  --pred_path "predicted_num_clusters_results/FINAL_${DATASET}.json" \
  --embed_method "finetuned" \
  --scale "${SCALE}"

cat "predicted_num_clusters_results/FINAL_${DATASET}.json"
