#!/usr/bin/env bash
set -euo pipefail

# 1. Define Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 2. Assign Arguments (Dynamic)
# $1: Dataset Name (e.g., banking77)
# $2: Model name/path (optional, defaults to instructor-large)
TASK_NAME="${1:-"banking77"}"
MODEL_NAME="${2:-"hkunlp/instructor-large"}"

# 3. Dynamic File Paths based on Task Name
DATA_PATH="${REPO_ROOT}/datasets/${TASK_NAME}/test.jsonl"
RESULT_FILE="${REPO_ROOT}/datasets/${TASK_NAME}/embeddings_finetuned.h5"

# Safety Check: Ensure data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

echo "----------------------------------------------------------------"
echo "🚀 EXTRACTING EMBEDDINGS"
echo "Dataset : $TASK_NAME"
echo "Model   : $MODEL_NAME"
echo "Output  : $RESULT_FILE"
echo "----------------------------------------------------------------"

# 4. Execute Python Script
python "${PARENT_DIR}/get_embedding.py" \
    --task_name "$TASK_NAME" \
    --data_path "$DATA_PATH" \
    --result_file "$RESULT_FILE" \
    --model_name "$MODEL_NAME" \
    --batch_size 64 \
    --measure \
    --overwrite

echo "✅ Embedding extraction complete."
