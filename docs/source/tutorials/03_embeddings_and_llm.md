# Embeddings and Local LLM Oracle

The core of ClusterLLM-Local is the use of local open-weight models to guide a smaller embedding model.

## Local LLM Architecture (Dual-LLM Strategy)

To optimize performance and reasoning quality, we use a two-model configuration served via Ollama:

* **Perspective Stage (Triplets)**: Uses `qwen2.5:3b`. A smaller model is efficient and sufficient for triplet comparisons.


* **Granularity Stage (Pairs)**: Uses `llama3.3:70b-instruct-q2_K`. A larger model provides the necessary reasoning for cross-cluster pairwise judgments.



## VRAM Management

Running multiple models and fine-tuning an embedder on a single GPU requires strict memory management.
The pipeline automatically unloads Ollama models (`keep_alive=0`) before starting the fine-tuning process to prevent Out-of-Memory (OOM) errors.

## Fine-Tuning Objective

We employ a stabilized triplet fine-tuning objective:

* 
**Anti-Collapse**: VICReg-inspired variance regularization.


* 
**Spike Damping**: EMA-based controller to mitigate noise in LLM supervision.



---
