# ClusterLLM-Local: Privacy-Preserving Text Clustering via Quantized Local LLMs

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ed)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-orange)]()

> **Note:** This is a **research adaptation** of the framework *ClusterLLM* (Zhang et al., 2023). It replaces proprietary APIs (GPT-4) with **local, quantized LLMs  to enable zero-cost, privacy-compliant clustering on consumer hardware (NVIDIA L4 minimum recommended) .

---

## Abstract & Motivation

Text clustering often lacks user-specified granularity and perspective. While **ClusterLLM** [1] solved this by using LLMs as "guides" for small embedders (like `Instructor-XL`), it relies on costly and closed-source APIs (OpenAI), raising data privacy concerns for sensitive domains (healthcare, finance) [1, 2].

**ClusterLLM-Local** democratizes this approach by:
1.  **Replacing the Oracle:** Switching from GPT-3.5/4 to **Ollama**.
2.  **Ensuring Privacy:** No data leaves the local execution environment (GDPR compliant).
3.  **Reducing Cost:** Moving from ~$0.6 per dataset [1] to **$0.00**.

### Comparison: Original vs. This Repo

| Feature | Original Paper (ClusterLLM) [1] | **ClusterLLM-Local (Ours)** |
| :--- | :--- | :--- |
| **LLM Backend** | GPT-3.5 / GPT-4 (API) | **Llama-3 / Mistral, etc (Local)** |
| **Privacy** | ❌ Data sent to OpenAI | ✅ **100% Offline / Private** |
| **Cost** | ~$0.6 per run | **$0.00** (Consumer GPU) |
| **Hardware** | CPU (API calls) | Single GPU (24GB VRAM recommended) |
| **Reproducibility**| Scripts | **Docker + DVC + Hydra** |

---
## Cross__Val__Strategy

```markdown 
| Dataset Category | Dataset Name (Large-Scale) | Optimal Stage 1 (Triplet Task) | Optimal Stage 2 (Pairwise Task) | Auxiliary/Baseline Models (Other LLMs) | Selection Rationale |
| --- | --- | --- | --- | --- | --- |
| **Intent Discovery** | Bank77, CLINC(I), MTOP(I), Massive(I)  | **deepseek-r1:32b** | **qwen2.5:32b** | llama3.1:8b-instruct-q8_0, llama3.2:3b-instruct-q8_0 | DeepSeek-R1 handles fine-grained intent logic; Qwen excels at pairwise consistency. |
| --- | --- | --- | --- | --- | --- |
| **Type Discovery** | FewRel, FewNerd, FewEvent | **qwen2.5:32b** | **llama3.3:70b-instruct-q2_K** | mixtral:8x7b-instruct-v0.1-q4_0, llama3:latest | High-parameter models are required to capture nuanced entity and relation types. |
| --- | --- | --- | --- | --- | --- |
| **Topic Mining** | StackEx, ArxivS2S, Reddit | **llama3.3:70b-instruct-q2_K** | **llama3.3:70b-instruct-q2_K** | qwen2.5:7b, gemma:7b-instruct-q4_K_M | Broad knowledge models are necessary for clustering complex academic and social topics. |
| --- | --- | --- | --- | --- | --- |
| **Emotion Detection** | GoEmo | **deepseek-r1:32b** | **qwen2.5:32b** | mistral:7b-instruct-q4_K_M, llama3:8b-instruct-q4_K_M | Reasoning-based models are better at distinguishing subtle emotional variances. |
| --- | --- | --- | --- | --- | --- |
| **Domain Discovery** | CLINC(D), MTOP(D), Massive(D) | **mixtral:8x7b-instruct-v0.1-q4_0** | **llama3.3:70b-instruct-q2_K** | llama3.1:8b-instruct-q8_0, gemma:7b-instruct-q4_K_M | Large models prevent over-segmentation in coarse-grained domain clustering. |
| --- | --- | --- | --- | --- | --- |
```

## Getting Started
### Prerequisites

* **NVIDIA GPU** (Minimum  24GB VRAM recommended).
* **Docker** & **NVIDIA Container Toolkit**.
* **Ollama** running locally (`ollama serve`).

### 1. Installation

Clone the repo and install dependencies via Poetry or Docker:

```bash
git clone [https://github.com/gackouhamady/clusterllm-local.git](https://github.com/gackouhamady/clusterllm-local.git)
cd clusterllm-local
make install

```

### 2. Data Setup (DVC)

We use DVC to version control datasets (Bank77, FewRel, etc.) and cached embeddings.
```bash
## Prepare data   from  source ( HF)

chmod +x scripts/run_prepare_data_only.sh

# single 
ALLOW_REMOTE_CODE=1 scripts/run_prepare_data_only.sh mtop_intent small
## All small

ALLOW_REMOTE_CODE=1 scripts/run_prepare_data_only.sh all small 42

## All large
ALLOW_REMOTE_CODE=1 scripts/run_prepare_data_only.sh all large 42
```
##  Prepare Push Pull Sync
```bash
chmod +x scripts/dvc_prepare_and_sync.sh
ALLOW_REMOTE_CODE=1 scripts/dvc_prepare_and_sync.sh push mtop_intent small 42
ALLOW_REMOTE_CODE=1 scripts/dvc_prepare_and_sync.sh push all small 42
ALLOW_REMOTE_CODE=1 scripts/dvc_prepare_and_sync.sh push all small 42
scripts/dvc_prepare_and_sync.sh pull all small
```
```bash
# Push data to remote storage (Drive/S3)
dvc push
```

```bash
# Pull data from remote storage (Drive/S3)
dvc pull
```




# Run  :
## bash  
```bash
# Full pipeline (end-to-end) in ONE command
# Args: <dataset> <scale> <llm_triplet> <llm_pairs> [seed]
bash scripts/run_full_pipeline_one_dataset_two_llms.sh clinc150 small deepseek-r1:32b qwen2.5:32b 42

 
```

To align with your preference for **`dvc repro`** while maintaining the **Poetry** environment and the parameters for the CLUSTERLLM framework, here is the updated command set.

Note that `dvc repro` does not natively support the `--queue` or `-S` (parameter override) flags found in `dvc exp run`. To change parameters using `repro`, you typically modify the `params.yaml` file or use a script to inject them.

### 1. Launch a Single Run

This command executes the pipeline specifically for the `full_pipeline_2llms` stage as defined in your DVC project.

```bash
# Run a single reproduction through Poetry
# Note: Ensure your params.yaml is set to clinc150/small/llama3.2/qwen2.5/42 first
poetry run dvc repro full_pipeline_2llms

```

### 2. Run a Grid of 100 Configurations

Since `dvc repro` executes immediately and doesn't use the experiment queue, we use a `while` loop to run them sequentially. This ensures you can process all 14 datasets and various granularities mentioned in the research.

**Sequential Execution Loop**
This script updates the configuration and runs the reproduction one after another.

```bash
# Run 100 reproductions sequentially
while read -r ds sc lt lp seed; do
  echo "Running: $ds $sc with $lt and $lp"
  
  # Optional: You can use 'yq' or 'sed' to update params.yaml here 
  # to ensure dvc repro uses the new variables
  
  poetry run dvc repro full_pipeline_2llms
done < configs/grid_100.txt

```

---

###  Pipeline Logic & Memory Constraints

* 
**Stage 1: Improving Perspective**: The pipeline uses triplet tasks (e.g., `<does A better correspond to B than C>`) to fine-tune the "small" embedder (Instructor).
* 
**Stage 2: Determining Granularity**: The framework then uses pairwise questions (e.g., `<do A and B belong to the same category>`) to find the best cluster scope.
* 
**VRAM Management**: Because we are calling two LLMs (DeepSeek and Qwen), sequential execution with `dvc repro` is actually safer for our **NVIDIA L4 (24GB)**. Running multiple 32B models in parallel would exceed the available memory, as these models are significantly larger than the "small" embedders used in the initial stages.

---

##  Methodology

The framework operates in two stages, adapted from [1]:

### Stage 1: Perspective (Triplet Task)

We fine-tune a small embedder (`Instructor-large`) using "hard triplets" generated by the LLM.

* **Prompt:** *"Select the example that better corresponds with the Query in terms of [Intent/Topic]..."*
* **Local Adaptation:** The `src/llm_client/ollama.py` module handles prompt formatting to ensure Llms outputs strict JSON/Indices compatible with the original logic.

### Stage 2: Granularity (Pairwise Task)

We determine the optimal number of clusters () by asking the LLM to verify if pairs of documents belong to the same cluster.

* **Efficiency:** Uses hierarchical clustering consistency scores to find the best cut [1].

---

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
