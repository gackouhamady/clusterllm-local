# src/clusterllm/llm_client/ollama.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests


MODEL_REGISTRY = {
    # déjà installés chez toi
    "llama3_q4km": "llama3:8b-instruct-q4_K_M",
    "mistral_q4km": "mistral:7b-instruct-q4_K_M",
    "gemma_q4km": "gemma:7b-instruct-q4_K_M",
    # ajoute tes 3 autres modèles ici si tu veux
    # "llama31_8b_q8": "llama3.1:8b-instruct-q8_0",
    # "mixtral_8x7b_q4km": "mixtral:8x7b-instruct-q4_K_M",
    # "qwen25_7b_q5": "qwen2.5:7b-instruct-q5_K_M",
}


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    timeout: int = 120

    # on supporte 2 façons d'initialiser le modèle
    model_key: Optional[str] = None
    model_tag: Optional[str] = None

    def __post_init__(self) -> None:
        if self.model_tag:
            self.model = self.model_tag
            return

        if self.model_key:
            if self.model_key not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown model_key='{self.model_key}'. "
                    f"Valid keys: {sorted(MODEL_REGISTRY.keys())}"
                )
            self.model = MODEL_REGISTRY[self.model_key]
            return

        # défaut raisonnable
        self.model = MODEL_REGISTRY["llama3_q4km"]

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]
