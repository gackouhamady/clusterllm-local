import argparse
import os
import time
from typing import Any, Dict, Optional, Tuple

from clusterllm.llm_client.ollama import OllamaClient


def build_ollama_client_from_args(args: Optional[argparse.Namespace] = None) -> OllamaClient:
    """
    Build a LOCAL Ollama client.
    Priority:
      1) CLI args
      2) Environment variables
      3) Safe defaults
    """
    base_url = None
    model_key = None
    timeout = None
    temperature = None
    num_predict = None

    if args is not None:
        base_url = getattr(args, "ollama_base_url", None)
        model_key = getattr(args, "ollama_model", None)
        timeout = getattr(args, "ollama_timeout", None)
        temperature = getattr(args, "ollama_temperature", None)
        num_predict = getattr(args, "ollama_num_predict", None)

    base_url = base_url or os.getenv("CLUSTERLLM_OLLAMA_BASE_URL", "http://localhost:11434")
    model_key = model_key or os.getenv("CLUSTERLLM_OLLAMA_MODEL_KEY", "mistral:7b-instruct-q4_K_M")
    timeout = int(timeout or os.getenv("CLUSTERLLM_OLLAMA_TIMEOUT", "120"))
    temperature = float(temperature or os.getenv("CLUSTERLLM_OLLAMA_TEMPERATURE", "0.0"))
    num_predict = int(num_predict or os.getenv("CLUSTERLLM_OLLAMA_NUM_PREDICT", "64"))

    options = {
        "temperature": temperature,
        "num_predict": num_predict,
    }

    return OllamaClient(
        base_url=base_url,
        timeout=timeout,
        model_key=model_key,
        options=options,
    )


def delayed_completion(
    client: OllamaClient,
    prompt: str,
    delay_in_seconds: float = 1.0,
    max_trials: int = 1,
) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    """
    LOCAL delayed completion using Ollama.
    Returns an OpenAI-compatible structure:
      {"choices": [{"message": {"content": text}}]}
    """
    time.sleep(delay_in_seconds)

    output: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None

    for _ in range(max_trials):
        try:
            text = client.generate(prompt)
            output = {"choices": [{"message": {"content": text}}]}
            error = None
            break
        except Exception as e:
            error = e

    return output, error


def post_process(completion: Dict[str, Any]):
    content = completion["choices"][0]["message"]["content"].strip()
    result = []

    if "Yes" in content and "No" not in content:
        result.append("Yes")
    elif "No" in content and "Yes" not in content:
        result.append("No")

    return content, result


def add_ollama_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    CLI arguments to select the LOCAL LLM (lightweight model).
    """
    parser.add_argument("--ollama-base-url", type=str, default=None, help="Ollama server URL")
    parser.add_argument("--ollama-model", type=str, default=None, help="Ollama model tag")
    parser.add_argument("--ollama-timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--ollama-temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--ollama-num-predict", type=int, default=None, help="Max generated tokens")
    return parser
