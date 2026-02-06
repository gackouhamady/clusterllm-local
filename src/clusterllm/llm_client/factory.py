import os

from clusterllm.llm_client.ollama import OllamaClient


def get_llm_client() -> OllamaClient:
    """
    Central place to configure the local LLM backend.
    You control it via environment variables:

    - CLUSTERLLM_OLLAMA_BASE_URL (default: http://localhost:11434)
    - CLUSTERLLM_OLLAMA_MODEL_KEY (default: llama3_q4km)
    - CLUSTERLLM_OLLAMA_TIMEOUT (default: 120)
    """
    base_url = os.getenv("CLUSTERLLM_OLLAMA_BASE_URL", "http://localhost:11434")
    model_key = os.getenv("CLUSTERLLM_OLLAMA_MODEL_KEY", "llama3_q4km")
    timeout = int(os.getenv("CLUSTERLLM_OLLAMA_TIMEOUT", "120"))

    # options Ollama (safe defaults for classification Yes/No)
    options = {
        "temperature": float(os.getenv("CLUSTERLLM_OLLAMA_TEMPERATURE", "0.0")),
        "num_predict": int(os.getenv("CLUSTERLLM_OLLAMA_NUM_PREDICT", "64")),
    }

    return OllamaClient(
        base_url=base_url,
        timeout=timeout,
        model_key=model_key,
        options=options,
    )
