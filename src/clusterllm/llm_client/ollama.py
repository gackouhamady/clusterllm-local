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
        
        # On définit des options par défaut, tout en permettant de les surcharger via kwargs
        self.options = kwargs.get('options', {
            "temperature": 0.0,  # 0.0 est recommandé pour la consistance des triplets/paires
            "num_predict": 64    # Augmenté pour éviter les réponses tronquées
        })

    def generate(self, prompt, model=None):
        """
        Méthode principale utilisée par predict_triplet.py et predict_pairs.py.
        L'argument 'model' est maintenant accepté pour éviter l'erreur TypeError.
        """
        # Utilise le modèle passé en argument, sinon celui défini à l'initialisation
        target_model = model if model else self.model_name
        
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": self.options
        }
        
        try:
            response = requests.post(
                self.base_url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Récupération du texte de la réponse
            result_text = response.json().get("response", "").strip()
            return result_text
            
        except Exception as e:
            # On lève l'exception pour que delayed_completion (dans tools.py) 
            # puisse capturer l'erreur et gérer les tentatives (max_trials)
            print(f"❌ Erreur Ollama (Model: {target_model}): {e}")
            raise e