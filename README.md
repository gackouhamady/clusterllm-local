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


## Getting Started
### Prerequisites

* **NVIDIA GPU** (Minimum  24GB VRAM recommended).
* **Docker** & **NVIDIA Container Toolkit**.
* **Ollama** running locally (`ollama serve`).

### 1. Installation

Clone the repo and install dependencies via Poetry or Docker:

```bash
git clone https://github.com/gackouhamady/clusterllm-local.git
cd clusterllm-local
make install

```

### 2. DVC

#### Run first data pipelines   among  differentes configurations
##### Préparation des datasets (Small)

```bash
poetry run dvc repro prepare_small@bank77
poetry run dvc repro prepare_small@clinc150
poetry run dvc repro prepare_small@massive_intent
poetry run dvc repro prepare_small@massive_domain
poetry run dvc repro prepare_small@mtop_intent
poetry run dvc repro prepare_small@mtop_domain
poetry run dvc repro prepare_small@go_emotions
poetry run dvc repro prepare_small@reddit
poetry run dvc repro prepare_small@stackex
poetry run dvc repro prepare_small@arxiv
poetry run dvc repro prepare_small@few_event
poetry run dvc repro prepare_small@few_nerd_nat
poetry run dvc repro prepare_small@few_rel_nat
poetry run dvc repro prepare_small@clinc_intent
poetry run dvc repro prepare_small@clinc_domain

```

##### Préparation des datasets (Large)

```bash
poetry run dvc repro prepare_large@bank77
poetry run dvc repro prepare_large@clinc150
poetry run dvc repro prepare_large@massive_intent
poetry run dvc repro prepare_large@massive_domain
poetry run dvc repro prepare_large@mtop_intent
poetry run dvc repro prepare_large@mtop_domain
poetry run dvc repro prepare_large@go_emotions
poetry run dvc repro prepare_large@reddit
poetry run dvc repro prepare_large@stackex
poetry run dvc repro prepare_large@arxiv
poetry run dvc repro prepare_large@few_event
poetry run dvc repro prepare_large@few_nerd_nat
poetry run dvc repro prepare_large@few_rel_nat
poetry run dvc repro prepare_large@clinc_intent
poetry run dvc repro prepare_large@clinc_domain
```

- all datasets prepare (small/large) :
 ```bash
poetry  run dvc repro prepare_small
poetry run dvc repro prepare_large
 ```
```bash
# Push data to remote storage (Drive/S3)
dvc push
```
```bash
# Pull data from remote storage (Drive/S3)
dvc pull
```

#### Run Models Pipelines ( contributions  / faithful reproduction) :


##### Pipeline de contribution (full_pipeline_2llms_cont)
```bash
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=bank77"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=clinc150"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=massive_intent"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=massive_domain"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=mtop_intent"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=mtop_domain"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=go_emotions"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=reddit"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=stackex"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=arxiv"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=few_event"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=few_nerd_nat"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=few_rel_nat"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=clinc_intent"
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=clinc_domain"
```
##### Pipeline réel (full_pipeline_2llms)
```bash
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=bank77"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=clinc150"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=massive_intent"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=massive_domain"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=mtop_intent"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=mtop_domain"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=go_emotions"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=reddit"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=stackex"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=arxiv"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=few_event"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=few_nerd_nat"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=few_rel_nat"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=clinc_intent"
poetry run dvc exp run full_pipeline_2llms -S "run2llms.dataset=clinc_domain"
```
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
