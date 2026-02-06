#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  "llama3.1:8b-instruct-q8_0"
  "mixtral:8x7b-instruct-v0.1-q4_0"
  "qwen2.5:7b"
  "mistral:7b-instruct-q4_K_M"
  "gemma:7b-instruct-q4_K_M"
  "llama3:8b-instruct-q4_K_M"
)

echo "==> Ollama version"
ollama --version

echo "==> Pulling models..."
for m in "${MODELS[@]}"; do
  echo "----> ollama pull $m"
  ollama pull "$m"
done

echo "==> Installed models:"
ollama list
