import argparse
import os
import time
import json
from typing import Any, Dict, List, Optional, Tuple

from clusterllm.llm_client.ollama import OllamaClient


def build_ollama_client_from_args(args: Optional[argparse.Namespace] = None) -> OllamaClient:
    """
    Build an OllamaClient strictly for LOCAL inference.
    Priority: CLI args > environment variables > defaults.
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

    # Fallback to ENV or Defaults
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
    delay_in_seconds: float = 0.0,
    max_trials: int = 5,
    prompt: str = "",
    model: Optional[str] = None, # AJOUTÉ : Pour corriger l'erreur TypeError
) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    """
    Strict LOCAL completion: calls OllamaClient.generate(prompt) and returns an OpenAI-like dict.
    Accepts 'model' argument to match predict.py caller.
    """
    output: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None

    for trial in range(max_trials):
        try:
            if delay_in_seconds > 0:
                time.sleep(delay_in_seconds)
            
            # Utilise le modèle passé ou celui par défaut du client
            target_model = model or client.model_key
            
            # Appel à l'API locale Ollama
            text = client.generate(prompt=prompt, model=target_model)
            
            # Formatage compatible avec le reste du framework
            output = {"choices": [{"message": {"content": text}}]}
            error = None
            break
        except Exception as e:
            error = e
            if trial < max_trials - 1:
                time.sleep(2) # Attente avant nouvel essai en cas d'erreur réseau

    return output, error


def prepare_data(prompt: str, datum: Dict[str, Any]) -> str:
    """
    Constructs the triplet task prompt as described in the paper.
    Includes the instruction to respond without explanation.
    """
    postfix = "\n\nPlease respond with 'Choice 1' or 'Choice 2' without explanation."
    input_txt = datum["input"]
    
    # Nettoyage du tag 'Choice' s'il est déjà présent en fin de chaîne
    if input_txt.endswith("\nChoice"):
        input_txt = input_txt[:-7]
        
    return f"{prompt}\n\n{input_txt}{postfix}"


def post_process(completion: Dict[str, Any], choices: List[str]) -> Tuple[str, List[str]]:
    """
    Extracts the predicted choice from the LLM content.
    """
    content = completion["choices"][0]["message"]["content"].strip()
    result: List[str] = []
    
    # Normalisation pour la recherche (ex: "Choice 1")
    for choice in choices:
        # On vérifie si la réponse contient "Choice 1" ou "Choice 2"
        if choice.strip() in content:
            result.append(choice)
            
    return content, result


def add_ollama_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add CLI flags for local Ollama configuration.
    """
    parser.add_argument("--ollama-base-url", type=str, default=None, help="Ollama server URL")
    parser.add_argument("--ollama-model", type=str, default=None, help="Ollama model tag (e.g., deepseek-r1:32b)")
    parser.add_argument("--ollama-timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--ollama-temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--ollama-num-predict", type=int, default=None, help="Max tokens to generate")
    return parser