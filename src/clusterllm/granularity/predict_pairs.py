import argparse
import json
import os
import re
import concurrent.futures
from tqdm import tqdm

from tools import (
    add_ollama_cli_args,
    build_ollama_client_from_args,
    delayed_completion,
)


def _coerce_prompt(prompt):
    """
    prompt can be:
      - str
      - list like [prefix] or [prefix, postfix]
      - dict with keys like {"prompt": "...", "prefix": "...", "postfix": "..."}
    Returns: (prefix: str, postfix: str|None)
    """
    if prompt is None:
        return "", None

    # dict format
    if isinstance(prompt, dict):
        prefix = prompt.get("prompt") or prompt.get("prefix") or ""
        postfix = prompt.get("postfix")
        return str(prefix), (str(postfix) if postfix is not None else None)

    # list format
    if isinstance(prompt, list):
        prefix = prompt[0] if len(prompt) > 0 else ""
        postfix = prompt[1] if len(prompt) > 1 else None
        return str(prefix), (str(postfix) if postfix is not None else None)

    # default: string/other
    return str(prompt), None


def _get_input_text(datum):
    """
    datum can be:
      - {"input": "..."}  (expected)
      - {"text": "..."}   (other pipeline parts)
      - {"sent1": "...", "sent2": "..."} or similar -> fallback join
    """
    if not isinstance(datum, dict):
        return str(datum)

    if "input" in datum and datum["input"] is not None:
        return str(datum["input"])
    if "text" in datum and datum["text"] is not None:
        return str(datum["text"])

    # common alternative fields
    for k in ("sentence", "query", "utterance"):
        if k in datum and datum[k] is not None:
            return str(datum[k])

    # last resort: stringify datum (avoid crash)
    return json.dumps(datum, ensure_ascii=False)


def prepare_data(task_prompt, datum):
    """
    Build final prompt for the LLM.
    Robust to prompt being str/list/dict and datum using different text keys.
    """
    # Force concise answer
    default_postfix = "\n\nPlease respond with ONLY 'Yes' or 'No'."

    prefix, postfix = _coerce_prompt(task_prompt)
    input_txt = _get_input_text(datum)

    # if prompt file already provides a postfix, keep it, otherwise our default
    postfix = postfix if postfix is not None else default_postfix

    # Safety: ensure everything is string
    return str(prefix) + "\n" + str(input_txt) + str(postfix)


def robust_post_process(completion):
    """
    Parser intelligent et TRES STRICT pour extraire la décision du LLM.
    Gère les retours type dict (API) ou string brute (Ollama).
    """
    # 1) Extract raw text
    if isinstance(completion, dict):
        content = completion.get("response", "") or completion.get("content", "")
    else:
        content = str(completion)

    content_clean = content.strip()
    
    # 2) Logique de décision stricte
    # On remplace la ponctuation par des espaces et on met tout en minuscules
    words = re.sub(r'[^a-zA-Z0-9]', ' ', content_clean.lower()).split()
    
    # Si la réponse est vide
    if not words:
        return content_clean, ["No"]

    # Le premier mot de la phrase générée doit être strictement "yes"
    if words[0] == "yes":
        return content_clean, ["Yes"]
    elif words[0] == "no":
        return content_clean, ["No"]

    # S'il commence par autre chose ("The answer is...", "I think..."), on refuse par sécurité absolue
    return content_clean, ["No"]


def predict(args):
    client = build_ollama_client_from_args(args)
    # Récupération sécurisée du nom du modèle passé dans le bash
    model_name = getattr(args, "ollama_model", "ollama")
    model_nick = model_name.replace(":", "_")

    prompt_file_name = os.path.basename(args.prompt_file).split(".")[0]
    pred_name = os.path.basename(args.data_path).replace(".json", f"-{model_nick}-{prompt_file_name}.json")
    pred_path = os.path.join("predicted_pair_results", pred_name)
    os.makedirs("predicted_pair_results", exist_ok=True)

    print(f"🚀 Inférence lancée. Résultats : {pred_path}")

    # Load prompts
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Robust prompt selection for ANY dataset:
    task_prompt = (
        prompts.get(args.dataset)
        or prompts.get(str(args.dataset).lower())
        or prompts.get("banking77")
        or next(iter(prompts.values()))
    )

    # Load data
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = raw_data.get("test_inputs") if isinstance(raw_data, dict) else raw_data
    if not isinstance(data, list):
        raise ValueError(f"Unexpected data format in {args.data_path}: expected list or dict with 'test_inputs'.")

    # ---- DEBUT DU MULTI-THREADING ----
    
    # Aperçu du premier prompt
    if len(data) > 0:
        print("\n--- PROMPT PREVIEW ---")
        print(prepare_data(task_prompt, data[0]))
        print("----------------------\n")

    # Fonction isolée pour traiter UNE paire
    def process_pair(idx):
        datum = data[idx]

        # Ne pas écraser si déjà fait
        if isinstance(datum, dict) and "prediction" in datum and not args.overwrite:
            return idx, True

        prompt = prepare_data(task_prompt, datum)

        # C'EST ICI QUE LE BUG A ETE CORRIGE : Ajout de model=model_name
        completion, error = delayed_completion(
            client=client,
            prompt=prompt,
            delay_in_seconds=float(args.delay),
            max_trials=int(args.max_trials),
            model=model_name
        )

        if completion is None:
            print(f"Error index {idx}: {error}")
            return idx, False

        content, results = robust_post_process(completion)

        # Rendre compatible
        if not isinstance(datum, dict):
            datum = {"input": _get_input_text(datum)}
            data[idx] = datum

        datum["res"] = content          # raw text (debug)
        datum["output"] = results[0]    # Strict "Yes"/"No"
        datum["prediction"] = results   # ["Yes"] or ["No"]

        return idx, True

    # Récupère la variable d'environnement ou utilise 4 par défaut
    max_workers = int(os.environ.get("OLLAMA_NUM_PARALLEL", 4))
    save_counter = 0

    print(f"🚀 Lancement des PAIRES ({model_name}) avec {max_workers} requêtes en parallèle...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumission de toutes les tâches
        futures = {executor.submit(process_pair, i): i for i in range(len(data))}
        
        # Récupération progressive
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
            idx = futures[future]
            success = future.result()

            if not success:
                continue

            save_counter += 1
            if save_counter >= args.save_every:
                final_output = {"test_inputs": data, "pairs": data}
                with open(pred_path, "w", encoding="utf-8") as f:
                    json.dump(final_output, f, indent=4, ensure_ascii=False)
                save_counter = 0

    # Sauvegarde finale et décompte
    yes_count = sum(1 for d in data if isinstance(d, dict) and d.get("output") == "Yes")
    
    final_output = {"test_inputs": data, "pairs": data}
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

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