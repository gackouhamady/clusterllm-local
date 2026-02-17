import requests
import json
import time

class OllamaClient:
    def __init__(self, model_name="qwen2.5:7b", base_url="http://localhost:11434", **kwargs):
        """
        Client universel pour Ollama corrigé pour ClusterLLM.
        """
        self.model_name = model_name
        self.base_url = f"{base_url.rstrip('/')}/api/generate"
        self.timeout = kwargs.get('timeout', 120)

    def generate(self, prompt):
        """
        Méthode principale utilisée par predict_pairs.py
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.5,
                "num_predict": 15  # Légèrement augmenté pour laisser Qwen respirer
            }
        }
        
        try:
            response = requests.post(
                self.base_url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            # On retourne la réponse brute nettoyée
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"❌ Erreur Ollama ({self.model_name}): {e}")
            return None
