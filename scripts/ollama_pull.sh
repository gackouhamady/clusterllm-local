#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Install + enable Ollama on boot + configure parallelism + pull models
# ------------------------------------------------------------

MODELS=(
  # --- Modèles Très Petits & Ultra Rapides (< 4B paramètres) ---
  "gemma2:2b"                   # Excellent ratio taille/performance, très fort en logique
  "llama3.2:1b-instruct-q8_0"   # La version ultra-légère de Llama 3.2
  "qwen2.5:1.5b"                # Très rapide, très bon respect des instructions
  "qwen2.5:3b"                  # Un cran au-dessus du 1.5B, rivalise avec Llama 3.2 3B
  "phi3.5:latest"               # 3.8B paramètres, développé par Microsoft, roi du raisonnement
  "deepseek-r1:1.5b"            # Distillé pour exceller en logique mathématique et raisonnement
  "smollm2:1.7b"                # Un modèle compact très récent et performant

  # --- Modèles Petits/Moyens (~7B - 8B) ---
  "llama3.2:3b-instruct-q8_0"
  "llama3.1:8b-instruct-q8_0"
  "llama3:8b-instruct-q4_K_M"
  "llama3:latest"
  "qwen2.5:7b"
  "gemma:7b-instruct-q4_K_M"
  "mistral:7b-instruct-q4_K_M"

  # --- Modèles Larges & Complexes (> 10B) ---
  "mixtral:8x7b-instruct-v0.1-q4_0"
  "qwen2.5:32b"
  "deepseek-r1:32b"
  "llama3.3:70b-instruct-q2_K"
)

# ---> MODIFIÉ ICI : Passage à 16 pour maximiser l'utilisation du GPU
PARALLEL=16
HOST="127.0.0.1:11434"   # mets "127.0.0.1:11434" si tu veux seulement local

need_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: run as root (sudo)." >&2
    exit 1
  fi
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }

need_root

echo "==> Checking systemd..."
if ! has_cmd systemctl; then
  echo "ERROR: systemctl not found (no systemd). This script targets systemd-based VMs." >&2
  exit 1
fi

echo "==> Installing Ollama if missing..."
if ! has_cmd ollama; then
  if ! has_cmd curl; then
    echo "==> Installing curl..."
    apt-get update -y
    apt-get install -y curl
  fi

  echo "==> Running official Ollama install script..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "==> Ollama binary:"
command -v ollama
echo "==> Ollama version:"
ollama --version || true

echo "==> Enabling & starting ollama service..."
systemctl daemon-reload || true
systemctl enable --now ollama

echo "==> Configuring Ollama service overrides (parallelism + host)..."
mkdir -p /etc/systemd/system/ollama.service.d

cat >/etc/systemd/system/ollama.service.d/override.conf <<EOF
[Service]
Environment="OLLAMA_NUM_PARALLEL=${PARALLEL}"
Environment="OLLAMA_HOST=${HOST}"
EOF

systemctl daemon-reload
systemctl restart ollama

echo "==> Waiting for Ollama to be ready..."
for i in {1..60}; do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "==> Pulling models (one by one)..."
for m in "${MODELS[@]}"; do
  echo "----> ollama pull $m"
  ollama pull "$m"
done

echo "==> Installed models:"
ollama list

echo "==> Done."
echo "    Service: systemctl status ollama --no-pager"
echo "    Parallel: OLLAMA_NUM_PARALLEL=${PARALLEL}"
echo "    Host:     OLLAMA_HOST=${HOST}"