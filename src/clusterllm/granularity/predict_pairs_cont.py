#!/usr/bin/env python3
"""
predict_pairs_cont.py

Pairwise prediction with Ollama (local LLM) for CLUSTERLLM-style granularity supervision.

Contribution #3 extension: uncertainty- and noise-aware supervision
------------------------------------------------------------------
This file ONLY EXTENDS the original behavior by adding:

1) Self-consistency sampling (num_responses = M):
   - Collect M independent LLM responses per pair.
   - Store per-item vote list in `votes`.

2) Uncertainty estimation from votes:
   - p_yes (empirical probability of "Yes")
   - binary entropy (numerically stable)
   - confidence weight: w = 1 - H/log(2)

3) Active re-query (optional):
   - If max(p_yes, 1-p_yes) < requery_threshold:
     query `requery_extra` additional responses and recompute stats.

Backward compatibility:
- If num_responses == 1 and requery is disabled (defaults), the script preserves original behavior:
  one completion per pair, and the same output fields are produced.
- This script never removes or renames original output fields; it only adds:
  votes, p_yes, entropy, weight.

Important constraints:
- Prompt construction logic is preserved.
- Input loading and final JSON save logic are preserved.
- Concurrency structure is preserved (thread pool over items).
"""

import argparse
import concurrent.futures
import inspect
import json
import os
import re
import threading
import math
from typing import Any, Dict, Optional, Tuple, List

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
# Contribution #3: uncertainty helpers
# -----------------------------
def _binary_entropy(p: float) -> float:
    """
    Numerically stable binary entropy in natural log:
      H(p) = -p log p - (1-p) log(1-p)
    """
    eps = 1e-12
    if p <= 0.0 or p >= 1.0:
        return 0.0
    p = min(max(p, eps), 1.0 - eps)
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


def _stats_from_votes(votes: List[str]) -> Tuple[float, float, float]:
    """
    From votes in {"Yes","No"} compute:
      p_yes, entropy, weight
    weight = 1 - entropy/log(2), clipped to [0,1]
    """
    if not votes:
        p_yes = 0.5
        ent = math.log(2.0)
        w = 0.0
        return p_yes, ent, w

    yes = sum(1 for v in votes if str(v).strip().lower() == "yes")
    total = len(votes)
    p_yes = yes / total
    ent = _binary_entropy(p_yes)
    denom = math.log(2.0)
    w = 1.0 - (ent / denom if denom > 0 else 0.0)
    if w < 0.0:
        w = 0.0
    elif w > 1.0:
        w = 1.0
    return p_yes, ent, w


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
        pass

    attempts = []

    attempts.append(dict(kwargs))

    if "model" in kwargs and "model_name" not in kwargs:
        a = dict(kwargs)
        a["model_name"] = a.pop("model")
        attempts.append(a)

    if "model" in kwargs and "ollama_model" not in kwargs:
        a = dict(kwargs)
        a["ollama_model"] = a.pop("model")
        attempts.append(a)

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
    ds_key = str(args.dataset)
    task_prompt = (
        prompts.get(ds_key)
        or prompts.get(ds_key.lower())
        or prompts.get(ds_key.replace("-", "_"))
        or prompts.get(ds_key.lower().replace("-", "_"))
    )
    if task_prompt is None:
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

    def _ensure_uncertainty_fields(datum: Dict[str, Any]) -> None:
        """
        If a previous run already produced 'prediction' but not the new uncertainty fields,
        derive votes/stats conservatively without extra queries.
        """
        if all(k in datum for k in ("votes", "p_yes", "entropy", "weight")):
            return
        if "prediction" in datum:
            # prediction is usually ["Yes"] or ["No"]
            pred = datum.get("prediction")
            if isinstance(pred, list) and pred:
                v = str(pred[0])
            else:
                v = str(pred)
            votes = ["Yes" if v.strip().lower() == "yes" else "No"]
            p_yes, ent, w = _stats_from_votes(votes)
            datum["votes"] = votes
            datum["p_yes"] = p_yes
            datum["entropy"] = ent
            datum["weight"] = w

    def process_one(idx: int) -> Tuple[int, bool, Optional[str]]:
        """
        Returns: (idx, ok, error_message)
        NEVER raises (so futures never crash the whole run).
        """
        try:
            datum = data[idx]

            if isinstance(datum, dict) and "prediction" in datum and not args.overwrite:
                _ensure_uncertainty_fields(datum)
                return idx, True, None

            prompt = prepare_data(task_prompt, datum)

            # Self-consistency: collect M votes, then optionally re-query extra votes if uncertain.
            M = max(1, int(getattr(args, "num_responses", 1) or 1))
            requery_threshold = float(getattr(args, "requery_threshold", 0.0) or 0.0)
            requery_extra = int(getattr(args, "requery_extra", 0) or 0)

            votes: List[str] = []
            first_raw_text: Optional[str] = None

            def _one_call() -> Tuple[Optional[str], Optional[str]]:
                completion, error = _call_delayed_completion_safe(
                    client=client,
                    prompt=prompt,
                    delay_in_seconds=float(args.delay),
                    max_trials=int(args.max_trials),
                    model=model_name,
                )
                if completion is None:
                    return None, str(error)
                raw_text, decision = strict_yes_no(completion)
                return (raw_text, decision)

            # Initial M responses
            for _ in range(M):
                raw_text, decision_or_err = _one_call()
                if raw_text is None:
                    return idx, False, f"{decision_or_err}"
                if first_raw_text is None:
                    first_raw_text = raw_text
                votes.append(decision_or_err)

            p_yes, ent, w = _stats_from_votes(votes)
            max_p = max(p_yes, 1.0 - p_yes)

            # Active re-query (optional; disabled by default)
            if requery_extra > 0 and 0.0 < requery_threshold < 1.0 and max_p < requery_threshold:
                for _ in range(requery_extra):
                    raw_text, decision_or_err = _one_call()
                    if raw_text is None:
                        return idx, False, f"{decision_or_err}"
                    votes.append(decision_or_err)
                p_yes, ent, w = _stats_from_votes(votes)

            # Preserve original output fields:
            # - content: keep first raw text
            # - output/prediction: use majority vote when M>1 (M=1 matches original)
            yes_votes = sum(1 for v in votes if str(v).strip().lower() == "yes")
            decision_final = "Yes" if yes_votes >= (len(votes) - yes_votes) else "No"

            if not isinstance(datum, dict):
                datum = {"input": _get_input_text(datum)}
                data[idx] = datum

            datum["prepared"] = prompt
            datum["content"] = first_raw_text if first_raw_text is not None else ""
            datum["output"] = decision_final
            datum["prediction"] = [decision_final]

            # Contribution #3 fields
            datum["votes"] = votes
            datum["p_yes"] = p_yes
            datum["entropy"] = ent
            datum["weight"] = w

            return idx, True, None

        except Exception as e:
            return idx, False, f"process_one exception: {repr(e)}"

    max_workers = int(os.environ.get("OLLAMA_NUM_PARALLEL", 4))
    print(f"🚀 Lancement des PAIRES ({model_name}) avec {max_workers} requêtes en parallèle...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one, i) for i in range(len(data))]

        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            idx, ok, err = fut.result()

            if not ok:
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

    # Contribution #3: self-consistency + re-query (optional; defaults preserve behavior)
    p.add_argument("--num_responses", type=int, default=1, help="Number of LLM responses per pair (self-consistency M).")
    p.add_argument(
        "--requery_threshold",
        type=float,
        default=0.0,
        help="If max(p_yes,1-p_yes) < threshold, query additional responses (0 disables).",
    )
    p.add_argument(
        "--requery_extra",
        type=int,
        default=0,
        help="Number of additional responses to query when uncertain (0 disables).",
    )

    add_ollama_cli_args(p)
    args = p.parse_args()
    predict(args)