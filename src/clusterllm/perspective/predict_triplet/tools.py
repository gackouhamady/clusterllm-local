import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from clusterllm.llm_client.ollama import OllamaClient


def build_ollama_client_from_args(args: Optional[argparse.Namespace] = None) -> OllamaClient:
    """
    Build an OllamaClient strictly for LOCAL inference.
    Priority: CLI args > environment variables > defaults.

    CLI:
      --ollama-base-url (default: http://localhost:11434)
      --ollama-model    (default: llama3:8b-instruct-q4_K_M)  # change to your lightest model
      --ollama-timeout  (default: 120)
      --ollama-temperature (default: 0.0)
      --ollama-num-predict (default: 64)

    ENV (fallback):
      CLUSTERLLM_OLLAMA_BASE_URL
      CLUSTERLLM_OLLAMA_MODEL_KEY
      CLUSTERLLM_OLLAMA_TIMEOUT
      CLUSTERLLM_OLLAMA_TEMPERATURE
      CLUSTERLLM_OLLAMA_NUM_PREDICT
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
    model_key = model_key or os.getenv("CLUSTERLLM_OLLAMA_MODEL_KEY", "llama3:8b-instruct-q4_K_M")
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
    delay_in_seconds: float = 1.0,
    max_trials: int = 1,
    prompt: str = "",
) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    """
    Strict LOCAL completion: calls OllamaClient.generate(prompt) and returns an OpenAI-like dict:
      {"choices":[{"message":{"content": "<text>"}}]}
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


def prepare_data(prompt: str, datum: Dict[str, Any]) -> str:
    postfix = "\n\nPlease respond with 'Choice 1' or 'Choice 2' without explanation."
    input_txt = datum["input"]
    if input_txt.endswith("\nChoice"):
        input_txt = input_txt[:-7]
    return prompt + input_txt + postfix


def post_process(completion: Dict[str, Any], choices: List[str]) -> Tuple[str, List[str]]:
    content = completion["choices"][0]["message"]["content"].strip()
    result: List[str] = []
    for choice in choices:
        choice_txt = "Choice" + choice
        if choice_txt in content:
            result.append(choice)
    return content, result


def add_ollama_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add CLI flags so you can choose the lightweight Ollama model at runtime.
    Example:
      --ollama-model llama3.1:8b-instruct-q8_0
      --ollama-model llama3:8b-instruct-q4_K_M
      --ollama-model mistral:7b-instruct-q4_K_M
    """
    parser.add_argument("--ollama-base-url", type=str, default=None, help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--ollama-model", type=str, default=None, help="Ollama model tag (e.g., llama3:8b-instruct-q4_K_M)")
    parser.add_argument("--ollama-timeout", type=int, default=None, help="Timeout in seconds (default: 120)")
    parser.add_argument("--ollama-temperature", type=float, default=None, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--ollama-num-predict", type=int, default=None, help="Max tokens to generate (default: 64)")
    return parser
