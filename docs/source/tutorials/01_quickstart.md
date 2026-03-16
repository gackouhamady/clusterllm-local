# Quickstart Guide

This guide helps you set up and run your first ClusterLLM-Local pipeline. ClusterLLM-Local is a fully local, reproducible re-implementation of the ClusterLLM framework for text clustering.

## Prerequisites

* 
**NVIDIA GPU**: Minimum 24GB VRAM recommended (e.g., NVIDIA L4).


* 
**Local LLM Server**: Ollama must be installed and running (`ollama serve`).


* 
**Software**: Docker, NVIDIA Container Toolkit, and Poetry.



## 1. Setup Environment

Clone the repository and install dependencies using Poetry:

```bash
git clone https://github.com/gackouhamady/clusterllm-local.git
cd clusterllm-local
poetry install

```

Alternatively, build the Docker image to ensure system-level portability:

```bash
docker build -f docker/Dockerfile -t clusterllm-local:core .

```

## 2. Model Preparation

Pull the recommended models through Ollama:

```bash
ollama pull qwen2.5:3b  # For Perspective Stage
ollama pull llama3.3:70b-instruct-q2_K  # For Granularity Stage

```

## 3. Run Your First Pipeline

Execute the full end-to-end pipeline for a sample dataset (e.g., Bank77) using DVC:

```bash
poetry run dvc exp run full_pipeline_2llms_cont -S "run2llms.dataset=bank77"

```

This command triggers baseline embedding extraction, triplet sampling, LLM supervision, and granularity estimation in one automated flow.

---
