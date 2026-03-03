#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

PREP="$ROOT/src/data/prepare_data.py"
DATA_DIR="$ROOT/src/clusterllm/datasets"

die() { echo "ERROR: $*" >&2; exit 1; }

ds="${1:-}"
sc="${2:-}"
seed_arg="${3:-}"

[[ -z "${ds}" || -z "${sc}" ]] && die "Usage: $0 <dataset> <small|large> [seed]"
[[ "$sc" == "small" || "$sc" == "large" ]] || die "scale must be small|large, got: $sc"

SEED="${SEED:-42}"
[[ -n "${seed_arg:-}" ]] && SEED="$seed_arg"

mkdir -p "$DATA_DIR/$ds"

tmp="$(mktemp -d)"
cleanup() { rm -rf "$tmp" || true; }
trap cleanup EXIT

TMP_DATASETS="$tmp/datasets"
mkdir -p "$TMP_DATASETS"

echo "==> PREPARE (isolated): ds=$ds scale=$sc seed=$SEED tmp=$TMP_DATASETS"

# generate BOTH in temp (prepare_data.py behavior)
python "$PREP" \
  --dataset "$ds" \
  --seed "$SEED" \
  --datasets-dir "$TMP_DATASETS"

src_file="$TMP_DATASETS/$ds/$sc.jsonl"
dst_file="$DATA_DIR/$ds/$sc.jsonl"

test -f "$src_file" || die "Missing temp output: $src_file"
cp -f "$src_file" "$dst_file"
test -f "$dst_file" || die "Copy failed: $dst_file"

echo "OK: $dst_file"