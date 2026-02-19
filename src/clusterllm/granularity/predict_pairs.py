import argparse
import json
import os
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
    Parser intelligent pour extraire la décision du LLM.
    Gère les retours type dict (API) ou string brute (Ollama).
    """
    # 1) Extract raw text
    if isinstance(completion, dict):
        content = completion.get("response", "") or completion.get("content", "")
    else:
        content = str(completion)

    content = content.strip()
    normalized = content.lower()

    # 2) Decision logic
    if "yes" in normalized:
        return content, ["Yes"]
    if "no" in normalized:
        return content, ["No"]

    # fallback
    return content, ["No"]


def predict(args):
    client = build_ollama_client_from_args(args)
    model_nick = getattr(args, "ollama_model", "ollama").replace(":", "_")

    prompt_file_name = os.path.basename(args.prompt_file).split(".")[0]
    pred_name = os.path.basename(args.data_path).replace(".json", f"-{model_nick}-{prompt_file_name}.json")
    pred_path = os.path.join("predicted_pair_results", pred_name)
    os.makedirs("predicted_pair_results", exist_ok=True)

    print(f"🚀 Inférence lancée. Résultats : {pred_path}")

    # Load prompts
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Robust prompt selection:
    # - try exact dataset key
    # - try lowercase
    # - common alias banking77
    # - otherwise first value
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

    yes_count = 0
    for idx, datum in tqdm(enumerate(data), total=len(data)):
        if isinstance(datum, dict) and "prediction" in datum and not args.overwrite:
            if "Yes" in datum.get("prediction", []):
                yes_count += 1
            continue

        prompt = prepare_data(task_prompt, datum)

        if idx == 0:
            print("\n--- PROMPT PREVIEW ---")
            print(prompt)
            print("----------------------\n")

        completion, error = delayed_completion(
            client=client,
            prompt=prompt,
            delay_in_seconds=float(args.delay),
            max_trials=int(args.max_trials),
        )

        if completion is None:
            print(f"Error index {idx}: {error}")
            continue

        content, results = robust_post_process(completion)

        # store results (compatible with predict_num_clusters.py)
        if not isinstance(datum, dict):
            # if datum isn't a dict, convert to dict so downstream code doesn't crash
            datum = {"input": _get_input_text(datum)}
            data[idx] = datum

        datum["res"] = content          # raw text (debug)
        datum["output"] = results[0]    # "Yes"/"No"
        datum["prediction"] = results   # ["Yes"] or ["No"]

        if results[0] == "Yes":
            yes_count += 1

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
