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


The following table provides a comprehensive cross-validation of our  local LLMs against the 14 large datasets used in the ClusterLLM project, optimized for Stage 1 (Perspective/Triplet Task) and Stage 2 (Granularity/Pairwise Task).

## Local LLM Cross-Validation for ClusterLLM Framework

| Dataset Category | Dataset Name (Large) | Stage 1: Perspective (Triplet Task) | Stage 2: Granularity (Pairwise Task) | Technical Rationale |
| --- | --- | --- | --- | --- |
| **Intent Discovery** | Bank77 | **deepseek-r1:32b** | **qwen2.5:32b** | R1’s chain-of-thought is optimal for disambiguating fine-grained banking intents. |
| **Intent Discovery** | CLINC(I) | **deepseek-r1:32b** | **qwen2.5:32b** | DeepSeek excels at logical separation of diverse intents across multiple domains. |
| **Intent Discovery** | MTOP(I) | **qwen2.5:32b** | **llama3.3:70b** | Qwen’s strong instruction following handles complex semantic parsing queries effectively. |
| **Intent Discovery** | Massive(I) | **deepseek-r1:32b** | **llama3.3:70b** | Massive requires high reasoning capacity to guide embedders through typo-prone user utterances. |
| **Type Discovery** | FewRel | **qwen2.5:32b** | **llama3.3:70b** | Relation type discovery benefits from Qwen’s balanced performance in structured data. |
| **Type Discovery** | FewNerd | **qwen2.5:32b** | **llama3.3:70b** | Entity type discovery requires the high-level knowledge found in larger parameter models. |
| **Type Discovery** | FewEvent | **deepseek-r1:32b** | **llama3.3:70b** | R1 is best suited for the logical complexity of event trigger correspondence. |
| **Topic Mining** | StackEx | **llama3.3:70b** | **llama3.3:70b** | Llama 3.3’s vast general knowledge is crucial for clustering scientific StackExchange topics. |
| **Topic Mining** | ArxivS2S | **llama3.3:70b** | **llama3.3:70b** | High-parameter models are necessary to identify nuanced academic research categories. |
| **Topic Mining** | Reddit | **llama3.3:70b** | **qwen2.5:32b** | Llama 3.3 understands community context better for broader topic mining. |
| **Emotion Detection** | GoEmo | **deepseek-r1:32b** | **qwen2.5:32b** | Reasoning models are essential for identifying subtle differences between similar emotions. |
| **Domain Discovery** | CLINC(D) | **mixtral:8x7b** | **llama3.3:70b** | Mixtral provides a solid baseline for coarse-grained domain separation. |
| **Domain Discovery** | MTOP(D) | **mixtral:8x7b** | **llama3.3:70b** | Llama 3.3 70b avoids over-segmentation in domain-level clustering hierarchies. |
| **Domain Discovery** | Massive(D) | **mixtral:8x7b** | **llama3.3:70b** | Coarse-grained domains require high-capacity models to maintain consistency. |

---

### Implementation Guidelines for Your Project

* **Stage 1 (Perspective)**: Use **deepseek-r1:32b** for tasks requiring high logical precision (Intent, Emotion) and **llama3.3:70b** for knowledge-intensive clustering (Science, Topics).
* **Stage 2 (Granularity)**: Deploy **llama3.3:70b** as the primary decision-maker for the number of clusters. Its superior instruction following ensures consistency between hierarchical levels and LLM predictions.
* **Baseline Calibration**: Use **llama3.2:3b-instruct-q8_0** or **llama3.1:8b-instruct-q8_0** to run fast, initial tests across all 14 datasets before committing to 32b/70b models.





## Getting Started

### Prerequisites

* **NVIDIA GPU** (Minimum 12GB VRAM, 24GB recommended for Llama-3-8B).
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
# Pull data from remote storage (Drive/S3)
dvc pull

```
# Run  :
# 1) Initial embeddings (Perspective)
```bash
cd ~/clusterllm-local/src/clusterllm/perspective/finetuning
bash scripts/get_embedding.sh
```
# 2) Triplet sampling

```bash
cd ~/clusterllm-local/src/clusterllm/perspective/predict_triplet
bash  scripts/triplet_sampling.sh
```
# 3) Triplet prediction (robust simulation: copy input to predicted)

```bash
cd ~/clusterllm-local/src/clusterllm/perspective/predict_triplet
bash  scripts/predict_triplet.sh
```
# 4) Convert triplets for finetuning
```bash
cd ~/clusterllm-local/src/clusterllm/perspective/finetuning
bash scripts/convert_triplet.sh
bash scripts/convert_triplet_self.sh
```

# 5) Finetune (create checkpoint)
```bash
cd ~/clusterllm-local/src/clusterllm/perspective/finetuning
bash  scripts/finetune.sh
```

# 6) Finetuned embeddings (for Granularity)

```bash
cd ~/clusterllm-local/src/clusterllm/perspective/finetuning
bash scripts/get_embedding.sh
```

# 9) Pair sampling (Granularity)
```bash
cd ~/clusterllm-local/src/clusterllm/granularity
bash scripts/sample_pairs.sh
```



```bash
cd ~/clusterllm-local/src/clusterllm/granularity
bash scripts/sample_pairs_for_prompt.sh
```

# 10) Predict pairs (Ollama) and final cluster count
```bash
cd ~/clusterllm-local/src/clusterllm/granularity
bash  scripts/predict_pairs.sh
```

```bash
cd ~/clusterllm-local/src/clusterllm/granularity
bash  scripts/predict_num_clusters.sh
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
