# ClusterLLM-Local: Privacy-Preserving Text Clustering via Quantized Local LLMs

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ed)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-orange)]()

> **Note:** This is a **research adaptation** of the framework *ClusterLLM* (Zhang et al., 2023). It replaces proprietary APIs (GPT-4) with **local, quantized LLMs (Llama-3, Mistral)** to enable zero-cost, privacy-compliant clustering on consumer hardware (e.g., NVIDIA L4/T4).

---

## Abstract & Motivation

Text clustering often lacks user-specified granularity and perspective. While **ClusterLLM** [1] solved this by using LLMs as "guides" for small embedders (like `Instructor-XL`), it relies on costly and closed-source APIs (OpenAI), raising data privacy concerns for sensitive domains (healthcare, finance) [1, 2].

**ClusterLLM-Local** democratizes this approach by:
1.  **Replacing the Oracle:** Switching from GPT-3.5/4 to **Ollama** (Llama-3-8B-Quantized).
2.  **Ensuring Privacy:** No data leaves the local execution environment (GDPR compliant).
3.  **Reducing Cost:** Moving from ~$0.6 per dataset [1] to **$0.00**.

### Comparison: Original vs. This Repo

| Feature | Original Paper (ClusterLLM) [1] | **ClusterLLM-Local (Ours)** |
| :--- | :--- | :--- |
| **LLM Backend** | GPT-3.5 / GPT-4 (API) | **Llama-3 / Mistral (Local)** |
| **Privacy** | ❌ Data sent to OpenAI | ✅ **100% Offline / Private** |
| **Cost** | ~$0.6 per run | **$0.00** (Consumer GPU) |
| **Hardware** | CPU (API calls) | Single GPU (24GB VRAM recommended) |
| **Reproducibility**| Scripts | **Docker + DVC + Hydra** |

---

## Architecture

This repository follows strict **MLOps standards** (`cookiecutter-data-science`).

```text
clusterllm-local/
├── configs/               # Hydra Configuration (YAML)
│   ├── model_gpt.yaml     # Original Baseline
│   └── model_local.yaml   # ✅ Our Contribution (Ollama)
├── data/                  # Managed by DVC (Not in Git)
├── docker/                # Reproducible Environment (CUDA)
├── src/
│   ├── clusterllm/        # Core Logic (Adapted from [1])
│   │   ├── perspective/   # Stage 1: Triplet Task
│   │   └── granularity/   # Stage 2: Pairwise Task
│   └── llm_client/        # ✅ Modular LLM Interface
│       ├── abstract.py
│       └── ollama.py      # Local Inference Logic
├── dvc.yaml               # Data Pipeline
└── Makefile               # Task Automation

```

---

## Getting Started

### Prerequisites

* **NVIDIA GPU** (Minimum 12GB VRAM, 24GB recommended for Llama-3-8B).
* **Docker** & **NVIDIA Container Toolkit**.
* **Ollama** running locally (`ollama serve`).

### 1. Installation

Clone the repo and install dependencies via Poetry or Docker:

```bash
git clone [https://github.com/YOUR_USERNAME/clusterllm-local.git](https://github.com/YOUR_USERNAME/clusterllm-local.git)
cd clusterllm-local
make install

```

### 2. Data Setup (DVC)

We use DVC to version control datasets (Bank77, FewRel, etc.) and cached embeddings.

```bash
# Pull data from remote storage (Drive/S3)
dvc pull

```

### 3. Running the Pipeline

You can switch between the original implementation (GPT) and the local version using Hydra configs.

**To run with Local LLM (Llama-3):**

```bash
# Make sure Ollama is running: ollama run llama3
make run_local

```

**To run the Baseline (Local Mode):**
# 0) Setup (run from repo root)
```bash
cd ~/clusterllm-local
export REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export DATASET="banking77"
export DATA_DIR="${REPO_ROOT}/datasets/${DATASET}"
export OLLAMA_MODEL="mistral_q4km"
export OLLAMA_URL="http://localhost:11434"
```

# 1) Create required directories

```bash
mkdir -p "${DATA_DIR}" \
  "${REPO_ROOT}/src/clusterllm/granularity/prompts" \
  "${REPO_ROOT}/src/clusterllm/granularity/sampled_pair_results" \
  "${REPO_ROOT}/src/clusterllm/granularity/predicted_pair_results" \
  "${REPO_ROOT}/src/clusterllm/granularity/predicted_num_clusters_results" \
  "${REPO_ROOT}/src/clusterllm/perspective/finetuning/checkpoints" \
  "${REPO_ROOT}/src/clusterllm/perspective/predict_triplet/sampled_triplet_results" \
  "${REPO_ROOT}/src/clusterllm/perspective/predict_triplet/predicted_triplet_results" \
  "${REPO_ROOT}/src/clusterllm/perspective/finetuning/converted_triplet_results"
```

# 2) Initial embeddings (Perspective)
```bash
cd "${REPO_ROOT}/src/clusterllm/perspective/finetuning"
python get_embedding.py \
  --task_name "${DATASET}" \
  --data_path "${DATA_DIR}/test.jsonl" \
  --result_file "${DATA_DIR}/embeddings.pkl" \
  --model_name "sentence-transformers/all-mpnet-base-v2" \
  --batch_size 32 \
  --overwrite
```

# 3) Triplet sampling

```bash
cd "${REPO_ROOT}/src/clusterllm/perspective/predict_triplet"
python triplet_sampling.py \
  --dataset "${DATASET}" \
  --data_path "${DATA_DIR}/test.jsonl" \
  --feat_path "${DATA_DIR}/embeddings.pkl" \
  --out_dir "sampled_triplet_results" \
  --k 5
```
# 4) Triplet prediction (robust simulation: copy input to predicted)

```bash
cd "${REPO_ROOT}/src/clusterllm/perspective/predict_triplet"
TRIPLET_INPUT=$(ls -t sampled_triplet_results/${DATASET}*.json | head -n 1)
cp "${TRIPLET_INPUT}" "predicted_triplet_results/$(basename "${TRIPLET_INPUT}")"
TRIPLET_PRED=$(ls -t predicted_triplet_results/${DATASET}*.json | head -n 1)
```
# 5) Convert triplets for finetuning
```bash
cd "${REPO_ROOT}/src/clusterllm/perspective/finetuning"
python convert_triplet.py \
  --dataset "${DATASET}" \
  --data_path "${REPO_ROOT}/src/clusterllm/perspective/predict_triplet/${TRIPLET_PRED}" \
  --out_dir "converted_triplet_results"
CONVERTED_FILE=$(ls -t converted_triplet_results/${DATASET}*.json | head -n 1)
```
# 6) Finetune (create checkpoint)
```bash
python finetune.py \
  --model_name "sentence-transformers/all-mpnet-base-v2" \
  --train_data "${PWD}/${CONVERTED_FILE}" \
  --output_dir "checkpoints/${DATASET}_finetuned" \
  --num_epochs 1 \
  --batch_size 16
```
# 7) Patch checkpoint config.json (avoid MPNet loading issues)
```bash
CHECKPOINT_DIR="${REPO_ROOT}/src/clusterllm/perspective/finetuning/checkpoints/${DATASET}_finetuned"
python3 - <<PY
import json, os
p = os.path.join("${CHECKPOINT_DIR}", "config.json")
if os.path.exists(p):
    d = json.load(open(p))
    d["model_type"] = "mpnet"
    json.dump(d, open(p, "w"))
    print("patched", p)
else:
    print("missing", p)
PY
```
# 8) Finetuned embeddings (for Granularity)
```bash
cd "${REPO_ROOT}/src/clusterllm/perspective/finetuning"
python get_embedding.py \
  --task_name "${DATASET}" \
  --data_path "${DATA_DIR}/test.jsonl" \
  --result_file "${DATA_DIR}/embeddings_finetuned.h5" \
  --model_name "${CHECKPOINT_DIR}" \
  --batch_size 32 \
  --overwrite
```
# 9) Pair sampling (Granularity)
```bash
cd "${REPO_ROOT}/src/clusterllm/granularity"
python sample_pairs.py \
  --dataset "${DATASET}" \
  --data_path "${DATA_DIR}/test.jsonl" \
  --feat_path "${DATA_DIR}/embeddings_finetuned.h5" \
  --scale "small" \
  --embed_method "finetuned" \
  --k 1 \
  --out_dir "sampled_pair_results" \
  --min_clusters 2 \
  --max_clusters 200 \
  --seed 100
PAIRS_INPUT=$(ls -t sampled_pair_results/${DATASET}*.json | head -n 1)
```
# 10) Predict pairs (Ollama) and final cluster count
```bash
cd "${REPO_ROOT}/src/clusterllm/granularity"
echo "{\"${DATASET}\": \"Are the following two sentences in the same cluster?\\nSentence 1: {text_a}\\nSentence 2: {text_b}\\nAnswer (Yes/No):\"}" \
  > prompts/pair_prediction.json
```
```bash
python predict_pairs.py \
  --dataset "${DATASET}" \
  --data_path "${PWD}/${PAIRS_INPUT}" \
  --prompt_file "prompts/pair_prediction.json" \
  --temperature 0.0 \
  --ollama-model "${OLLAMA_MODEL}"
```
```bash
PAIRS_PRED=$(ls -t predicted_pair_results/${DATASET}*.json | head -n 1)
```
```bash
python predict_num_clusters.py \
  --dataset "${DATASET}" \
  --data_path "${DATA_DIR}/test.jsonl" \
  --clustering_results "${PWD}/${PAIRS_PRED}" \
  --pred_path "predicted_num_clusters_results/FINAL_${DATASET}.json" \
  --embed_method "finetuned" \
  --scale "small"
```
```bash
cat predicted_num_clusters_results/FINAL_${DATASET}.json
```


---

##  Methodology

The framework operates in two stages, adapted from [1]:

### Stage 1: Perspective (Triplet Task)

We fine-tune a small embedder (`Instructor-large`) using "hard triplets" generated by the LLM.

* **Prompt:** *"Select the example that better corresponds with the Query in terms of [Intent/Topic]..."*
* **Local Adaptation:** The `src/llm_client/ollama.py` module handles prompt formatting to ensure Llama-3 outputs strict JSON/Indices compatible with the original logic.

### Stage 2: Granularity (Pairwise Task)

We determine the optimal number of clusters () by asking the LLM to verify if pairs of documents belong to the same cluster.

* **Efficiency:** Uses hierarchical clustering consistency scores to find the best cut [1].

---

## Roadmap & Status

This project is part of a Master's Thesis (Data Science & AI).

* [x] **Phase 1:** MLOps Architecture (Docker, DVC, Hydra).
* [ ] **Phase 2:** Baseline Reproduction (Bank77) using GPT-3.5.
* [ ] **Phase 3:** Local Implementation (Ollama/vLLM integration).
* [ ] **Phase 4:** Benchmarking (Cost vs. NMI Score).
* [ ] **Phase 5:** Publication (ArXiv Preprint / Workshop Submission).

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{zhang2023clusterllm,
  title={ClusterLLM: Large Language Models as a Guide for Text Clustering},
  author={Zhang, Yuwei and Wang, Zihan and Shang, Jingbo},
  journal={arXiv preprint arXiv:2305.14871},
  year={2023}
}

```

**Original Repository:** [zhang-yu-wei/ClusterLLM](https://github.com/zhang-yu-wei/ClusterLLM)

---

### 👤 Author

- **Hamady GACKOU** Master's Student in Machine Learning and  Data Science
- [https://www.linkedin.com/in/hamady-gackou]
