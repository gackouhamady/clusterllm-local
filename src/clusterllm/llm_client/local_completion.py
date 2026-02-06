# src/clusterllm/llm_client/local_completion.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from clusterllm.llm_client.ollama import OllamaClient


def delayed_completion(
    client: OllamaClient,
    prompt: str,
    delay_in_seconds: float = 1.0,
    max_trials: int = 1,
) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    """
    Local equivalent of an OpenAI chat completion call, but via Ollama.

    Returns a dict compatible with:
        completion['choices'][0]['message']['content']
    """
    time.sleep(delay_in_seconds)

    output: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None

    for _ in range(max_trials):
        try:
            text = client.generate(prompt)
            output = {
                "choices": [
                    {"message": {"content": text}}
                ]
            }
            error = None
            break
        except Exception as e:  # network, timeout, etc.
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
