#!/usr/bin/env python3
import argparse
import concurrent.futures
import inspect
import json
import os
import re
import threading
from typing import Any, Dict, Optional, Tuple

from tqdm import tqdm

from tools import (
    add_ollama_cli_args,
    build_ollama_client_from_args,
    delayed_completion,
)


# -----------------------------
# Prompt + input normalization
# -----------------------------
def _coerce_prompt(prompt: Any) -> Tuple[str, Optional[str]]:
    """
    prompt can be:
      - str
      - list like [prefix] or [prefix, postfix]
      - dict with keys like {"prompt": "...", "prefix": "...", "postfix": "..."}
    Returns: (prefix: str, postfix: str|None)
    """
    if prompt is None:
        return "", None

    if isinstance(prompt, dict):
        prefix = prompt.get("prompt") or prompt.get("prefix") or ""
        postfix = prompt.get("postfix")
        return str(prefix), (str(postfix) if postfix is not None else None)

    if isinstance(prompt, list):
        prefix = prompt[0] if len(prompt) > 0 else ""
        postfix = prompt[1] if len(prompt) > 1 else None
        return str(prefix), (str(postfix) if postfix is not None else None)

    return str(prompt), None


def _get_input_text(datum: Any) -> str:
    """
    datum can be:
      - {"input": "..."}  (expected)
      - {"text": "..."}   (other pipeline parts)
      - arbitrary dict -> json dump fallback
      - str
    """
    if not isinstance(datum, dict):
        return str(datum)

    for k in ("input", "text", "sentence", "query", "utterance"):
        if k in datum and datum[k] is not None:
            return str(datum[k])

    return json.dumps(datum, ensure_ascii=False)


def prepare_data(task_prompt: Any, datum: Any) -> str:
    """
    Build final prompt for the LLM.
    Robust to prompt being str/list/dict and datum using different text keys.
    """
    default_postfix = "\n\nPlease respond with ONLY 'Yes' or 'No' (no explanation)."
    prefix, postfix = _coerce_prompt(task_prompt)
    input_txt = _get_input_text(datum)
    postfix = postfix if postfix is not None else default_postfix

    # safety join
    prefix = prefix.strip()
    return f"{prefix}\n{input_txt}{postfix}"


# -----------------------------
# Completion parsing (STRICT)
# -----------------------------
_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

def _extract_text_from_completion(completion: Any) -> str:
    """
    Supports:
      - Ollama-like dict: {"response": "..."} or {"content": "..."}
      - OpenAI-like dict: {"choices":[{"message":{"content":"..."}}]}
      - raw string
      - anything -> str()
    """
    if completion is None:
        return ""

    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        # common keys
        for k in ("response", "content", "text", "message"):
            v = completion.get(k)
            if isinstance(v, str) and v.strip():
                return v

        # openai style
        choices = completion.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                msg = c0.get("message")
                if isinstance(msg, dict):
                    ct = msg.get("content")
                    if isinstance(ct, str):
                        return ct
                # sometimes "text"
                tx = c0.get("text")
                if isinstance(tx, str):
                    return tx

        return json.dumps(completion, ensure_ascii=False)

    return str(completion)


def strict_yes_no(completion: Any) -> Tuple[str, str]:
    """
    Returns (raw_text, decision) where decision is strictly "Yes" or "No".
    Rule:
      - take FIRST standalone token yes/no
      - if none found -> "No" (fail-closed)
    """
    raw = _extract_text_from_completion(completion).strip()
    if not raw:
        return raw, "No"

    m = _YESNO_RE.search(raw)
    if not m:
        return raw, "No"

    tok = m.group(1).lower()
    return raw, ("Yes" if tok == "yes" else "No")


# -----------------------------
# delayed_completion adapter
# -----------------------------
def _call_delayed_completion_safe(**kwargs) -> Tuple[Any, Optional[str]]:
    """
    Call delayed_completion but NEVER crash on unexpected kwargs.
    Strategy:
      1) If signature introspection works, filter kwargs to accepted parameters.
      2) If still TypeError, progressively drop known suspects.
    """
    try:
        sig = inspect.signature(delayed_completion)
        accepted = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return delayed_completion(**filtered)
    except (ValueError, TypeError):
        # ValueError: some builtins/compiled callables
        pass

    # Fallback: try a few common parameter name variants
    # (because your codebase has OpenAI-style & Ollama-style versions).
    attempts = []

    # as-is
    attempts.append(dict(kwargs))

    # rename model -> model_name
    if "model" in kwargs and "model_name" not in kwargs:
        a = dict(kwargs)
        a["model_name"] = a.pop("model")
        attempts.append(a)

    # rename model -> ollama_model
    if "model" in kwargs and "ollama_model" not in kwargs:
        a = dict(kwargs)
        a["ollama_model"] = a.pop("model")
        attempts.append(a)

    # remove model entirely
    if "model" in kwargs:
        a = dict(kwargs)
        a.pop("model", None)
        attempts.append(a)

    last_err = None
    for a in attempts:
        try:
            return delayed_completion(**a)
        except TypeError as e:
            last_err = str(e)
            continue

    return None, f"delayed_completion TypeError after retries: {last_err}"


# -----------------------------
# Main predict
# -----------------------------
def predict(args):
    client = build_ollama_client_from_args(args)

    # robust model name selection
    model_name = getattr(args, "ollama_model", None) or getattr(args, "model_name", None) or "ollama"
    model_nick = str(model_name).replace(":", "_")

    prompt_file_name = os.path.basename(args.prompt_file).split(".")[0]
    pred_name = os.path.basename(args.data_path).replace(".json", f"-{model_nick}-{prompt_file_name}.json")
    pred_path = os.path.join("predicted_pair_results", pred_name)
    os.makedirs("predicted_pair_results", exist_ok=True)

    print(f"🚀 Inférence lancée. Résultats : {pred_path}")

    # Load prompts
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Strong dataset prompt resolution:
    # - exact key
    # - lowercase key
    # - sometimes prompt file stores under "task" naming (e.g., banking77)
    ds_key = str(args.dataset)
    task_prompt = (
        prompts.get(ds_key)
        or prompts.get(ds_key.lower())
        or prompts.get(ds_key.replace("-", "_"))
        or prompts.get(ds_key.lower().replace("-", "_"))
    )
    if task_prompt is None:
        # last resort: if prompt file has one entry, use it; otherwise crash early (better than wrong prompt)
        if isinstance(prompts, dict) and len(prompts) == 1:
            task_prompt = next(iter(prompts.values()))
        else:
            raise KeyError(
                f"Prompt introuvable pour dataset='{args.dataset}' dans {args.prompt_file}. "
                f"Clés dispo (extrait): {list(prompts.keys())[:20]}"
            )

    # Load data
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = raw_data.get("test_inputs") if isinstance(raw_data, dict) else raw_data
    if not isinstance(data, list):
        raise ValueError(f"Unexpected data format in {args.data_path}: expected list or dict with 'test_inputs'.")

    # Preview prompt
    if data:
        print("\n--- PROMPT PREVIEW ---")
        print(prepare_data(task_prompt, data[0]))
        print("----------------------\n")

    # Thread-safe counters + periodic save from main thread
    lock = threading.Lock()
    completed = 0

    def process_one(idx: int) -> Tuple[int, bool, Optional[str]]:
        """
        Returns: (idx, ok, error_message)
        NEVER raises (so futures never crash the whole run).
        """
        try:
            datum = data[idx]

            if isinstance(datum, dict) and "prediction" in datum and not args.overwrite:
                return idx, True, None

            prompt = prepare_data(task_prompt, datum)

            completion, error = _call_delayed_completion_safe(
                client=client,
                prompt=prompt,
                delay_in_seconds=float(args.delay),
                max_trials=int(args.max_trials),
                model=model_name,  # safe adapter will filter/rename if needed
            )

            if completion is None:
                return idx, False, f"{error}"

            raw_text, decision = strict_yes_no(completion)

            if not isinstance(datum, dict):
                datum = {"input": _get_input_text(datum)}
                data[idx] = datum

            datum["prepared"] = prompt
            datum["content"] = raw_text
            datum["output"] = decision
            datum["prediction"] = [decision]

            return idx, True, None

        except Exception as e:
            return idx, False, f"process_one exception: {repr(e)}"

    max_workers = int(os.environ.get("OLLAMA_NUM_PARALLEL", 16))
    print(f"🚀 Lancement des PAIRES ({model_name}) avec {max_workers} requêtes en parallèle...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one, i) for i in range(len(data))]

        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            idx, ok, err = fut.result()  # safe: process_one never raises

            if not ok:
                # log minimal (avoid spam)
                print(f"[WARN] idx={idx} failed: {err}")

            with lock:
                completed += 1
                do_save = (completed % int(args.save_every) == 0)

            if do_save:
                final_output = {"test_inputs": data, "pairs": data}
                with open(pred_path, "w", encoding="utf-8") as f:
                    json.dump(final_output, f, indent=2, ensure_ascii=False)

    # Final save
    final_output = {"test_inputs": data, "pairs": data}
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    yes_count = sum(1 for d in data if isinstance(d, dict) and d.get("output") == "Yes")
    print(f"\n✅ Terminé ! Yes détectés : {yes_count}/{len(data)}")
    print(f"✅ Fichier: {pred_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--delay", type=float, default=0.01)
    p.add_argument("--max_trials", type=int, default=3)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--overwrite", action="store_true")

    add_ollama_cli_args(p)
    args = p.parse_args()
    predict(args)
