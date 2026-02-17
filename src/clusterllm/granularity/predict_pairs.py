import argparse
import json
import os
from tqdm import tqdm

from tools import (
    add_ollama_cli_args,
    build_ollama_client_from_args,
    delayed_completion,
)

def prepare_data(prompt, datum):
    # On force le modèle à être concis
    postfix = "\n\nPlease respond with ONLY 'Yes' or 'No'."
    input_txt = datum["input"]
    return prompt + "\n" + input_txt + postfix

def robust_post_process(completion):
    """
    Parser intelligent pour extraire la décision du LLM.
    Gère les retours type dict (API) ou string brute (Ollama).
    """
    # 1. Extraction du texte brut
    if isinstance(completion, dict):
        # Cas où c'est un format API complexe
        content = completion.get("response", "") or completion.get("content", "")
    else:
        # Cas où OllamaClient renvoie directement le string
        content = str(completion)

    content = content.strip()
    normalized = content.lower()
    
    # 2. Logique de décision (Priorité au 'Yes')
    # On cherche si 'yes' est présent sans être contredit par un 'no' plus fort
    if "yes" in normalized:
        return content, ["Yes"]
    elif "no" in normalized:
        return content, ["No"]
    
    # Par défaut si incompréhensible
    return content, ["No"]

def predict(args):
    client = build_ollama_client_from_args(args)
    model_nick = getattr(args, 'ollama_model', 'ollama').replace(":", "_")
    
    prompt_file_name = args.prompt_file.split("/")[-1].split(".")[0]
    pred_path = args.data_path.split("/")[-1].replace(".json", f"-{model_nick}-{prompt_file_name}.json")
    pred_path = os.path.join("predicted_pair_results", pred_path)
    os.makedirs("predicted_pair_results", exist_ok=True)

    print(f"🚀 Inférence lancée. Résultats : {pred_path}")

    with open(args.prompt_file, "r") as f:
        prompts = json.load(f)
        task_prompt = prompts[args.dataset]

    with open(args.data_path, "r") as f:
        raw_data = json.load(f)
        data = raw_data["test_inputs"] if isinstance(raw_data, dict) else raw_data

    yes_count = 0
    for idx, datum in tqdm(enumerate(data), total=len(data)):
        if "prediction" in datum and not args.overwrite:
            if "Yes" in datum["prediction"]: yes_count += 1
            continue

        prompt = prepare_data(task_prompt, datum)
        
        if idx == 0:
            print("\n--- PROMPT PREVIEW ---")
            print(prompt)
            print("----------------------\n")

        # Appel au LLM
        completion, error = delayed_completion(
            client=client,
            prompt=prompt,
            delay_in_seconds=float(args.delay),
            max_trials=int(args.max_trials),
        )

        if completion is None:
            print(f"Error index {idx}: {error}")
            continue

        # Utilisation du parser robuste
        content, results = robust_post_process(completion)
        
        # On stocke les résultats pour predict_num_clusters.py
        data[idx]["res"] = content          # Texte brut (pour debug)
        data[idx]["output"] = results[0]    # Format "Yes"/"No"
        data[idx]["prediction"] = results   # Format liste compatible

        if results[0] == "Yes":
            yes_count += 1

    # Sauvegarde finale
    final_output = {"test_inputs": data, "pairs": data}
    with open(pred_path, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\n✅ Terminé ! Yes détectés : {yes_count}/{len(data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--delay", type=float, default=0.01)
    parser.add_argument("--max_trials", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    
    add_ollama_cli_args(parser)
    args = parser.parse_args()
    predict(args)