"""
predict_cont.py

CLUSTERLLM-style triplet prediction with Ollama (local LLM).

Contribution #3 extension: uncertainty- and noise-aware supervision
------------------------------------------------------------------
This file EXTENDS the original behavior by adding:

1) Self-consistency sampling (num_responses = M):
   - Collect M independent LLM responses per triplet.
   - Store per-item vote list in `votes`.

2) Uncertainty estimation from votes:
   - p_choice1, p_choice2 (empirical probabilities)
   - binary entropy (numerically stable)
   - confidence weight: w = 1 - H/log(2)

3) Active re-query (optional):
   - If max(p_choice1, p_choice2) < requery_threshold:
     query `requery_extra` additional responses and recompute stats.

Backward compatibility:
- If num_responses == 1 and requery_extra == 0 (default), the script behaves like the original:
  exactly one completion per item, same `content` and `prediction` fields.
- The script never removes or renames existing output fields; it only adds:
  votes, p_choice1, p_choice2, entropy, weight.

Important constraints:
- Prompt template logic, sampling logic, batching logic, and file IO logic are preserved.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from tools import (
    add_ollama_cli_args,
    build_ollama_client_from_args,
    delayed_completion,
    prepare_data,
    post_process,
)


def _normalize_vote_label(prediction: Any, options: Sequence[Any]) -> Optional[str]:
    """
    Normalize a post-processed prediction into 'Choice 1' or 'Choice 2' when possible.

    This is designed to be robust to:
      - predictions already being 'Choice 1' / 'Choice 2'
      - numeric shortcuts ('1' / '2')
      - predictions equal to one of the provided options (options[0] / options[1])
      - minor casing/whitespace differences

    Returns:
      'Choice 1' or 'Choice 2' if a mapping is recognized, otherwise None.
    """
    if prediction is None:
        return None

    s = str(prediction).strip()
    if not s:
        return None

    s_low = s.lower()

    # Direct label matches
    if s_low in {"choice 1", "choice1", "option 1", "option1", "1", "a"}:
        return "Choice 1"
    if s_low in {"choice 2", "choice2", "option 2", "option2", "2", "b"}:
        return "Choice 2"

    # Substring matches (common LLM outputs)
    if "choice 1" in s_low or "option 1" in s_low:
        return "Choice 1"
    if "choice 2" in s_low or "option 2" in s_low:
        return "Choice 2"

    # Option equality match
    if len(options) >= 2:
        try:
            o0 = str(options[0]).strip()
            o1 = str(options[1]).strip()
            if s == o0:
                return "Choice 1"
            if s == o1:
                return "Choice 2"
        except Exception:
            pass

    return None


def _binary_entropy(p: float) -> float:
    """
    Numerically stable binary entropy (natural log):
        H(p) = -p log p - (1-p) log(1-p)

    Uses clamping to avoid log(0).
    """
    # Clamp p away from 0 and 1 for numerical stability
    eps = 1e-12
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 0.0
    p_clamped = min(max(p, eps), 1.0 - eps)
    return -p_clamped * math.log(p_clamped) - (1.0 - p_clamped) * math.log(1.0 - p_clamped)


def _compute_vote_stats(choice1_count: int, choice2_count: int) -> Tuple[float, float, float, float]:
    """
    Compute (p_choice1, p_choice2, entropy, weight) from counts.

    weight = 1 - entropy / log(2), clipped to [0, 1].
    """
    total = choice1_count + choice2_count
    if total <= 0:
        # Degenerate (should not happen with valid post_process)
        p1 = 0.5
        p2 = 0.5
        ent = math.log(2.0)
        w = 0.0
        return p1, p2, ent, w

    p1 = choice1_count / total
    p2 = 1.0 - p1
    ent = _binary_entropy(p1)
    denom = math.log(2.0)
    w = 1.0 - (ent / denom if denom > 0 else 0.0)
    # Clip for safety
    if w < 0.0:
        w = 0.0
    elif w > 1.0:
        w = 1.0
    return p1, p2, ent, w


def predict(args: argparse.Namespace) -> None:
    """
    Run triplet prediction with optional self-consistency and active re-query.

    Required output fields (added; never removes existing fields):
      - votes: List[str] of 'Choice 1'/'Choice 2' in the order collected
      - p_choice1: float
      - p_choice2: float
      - entropy: float  (natural log)
      - weight: float   (1 - entropy/log(2))
    """
    # ---- 1. Handle Model Name Compatibility ----
    # Ensure args.ollama_model is populated even if passed as --model_name
    if not args.ollama_model and args.model_name:
        args.ollama_model = args.model_name

    # ---- 2. Build Client ----
    client = build_ollama_client_from_args(args)

    # ---- 3. Resolve Input File ----
    # If data_path is not explicitly provided, find the first json in input_dir matching the dataset
    input_file = args.data_path
    if not input_file and args.input_dir:
        # Search for json files that contain the dataset name (e.g. "bank77")
        search_pattern = os.path.join(args.input_dir, f"*{args.dataset}*.json")
        candidates = glob.glob(search_pattern)
        if not candidates:
            # Fallback: take any json
            candidates = glob.glob(os.path.join(args.input_dir, "*.json"))

        if candidates:
            input_file = candidates[0]
            print(f"Found input file: {input_file}")
        else:
            raise FileNotFoundError(f"No input JSON file found in {args.input_dir} for dataset {args.dataset}")
    elif not input_file:
        raise ValueError("Either --data_path or --input_dir must be provided.")

    # ---- 4. Construct Output Path ----
    # Naming convention: {original_name}-{model}-pred.json
    model_tag = args.ollama_model or "ollama"
    # Remove characters that might mess up filenames (like :)
    safe_model_tag = model_tag.replace(":", "_")

    temp_tag = f"-temp{round(args.ollama_temperature, 1)}" if args.ollama_temperature and args.ollama_temperature > 0 else ""

    base_name = os.path.basename(input_file)
    output_filename = base_name.replace(".json", f"-{safe_model_tag}{temp_tag}-pred.json")

    # Use output_dir if provided, otherwise default to "predicted_triplet_results"
    output_dir = args.output_dir or "predicted_triplet_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(output_dir, output_filename)
    print("Save in:", pred_path)

    # ---- 5. Resume or Load Data ----
    if os.path.exists(pred_path):
        print("Resuming from existing prediction file...")
        with open(pred_path, "r") as f:
            data = json.load(f)
    else:
        with open(input_file, "r") as f:
            data = json.load(f)

    # ---- 6. Load Prompts ----
    # Ensure prompts.json is found relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_path = os.path.join(script_dir, "prompts.json")

    if not os.path.exists(prompts_path):
        # Fallback for current working directory
        prompts_path = "prompts.json"

    with open(prompts_path, "r") as f:
        prompts = json.load(f)
        if args.dataset not in prompts:
            raise ValueError(f"Dataset '{args.dataset}' not found in prompts.json")
        task_prompt = prompts[args.dataset]

    # Prepare prompt once
    for d in data:
        if "prepared" not in d:
            d["prepared"] = prepare_data(task_prompt, d)

    # ---- 7. Inference Loop (MULTI-THREADING) ----
    if len(data) > 0 and "prepared" in data[0]:
        print(f"Sample Prompt:\n{data[0]['prepared']}\n")

    def _ensure_uncertainty_fields(datum: Dict[str, Any]) -> None:
        """
        Ensure Contribution #3 output fields exist.
        If prediction already exists (from previous runs), derive fields without re-querying.
        """
        required = ("votes", "p_choice1", "p_choice2", "entropy", "weight")
        if all(k in datum for k in required):
            return

        # Derive from existing prediction if possible
        if "prediction" in datum and "options" in datum:
            label = _normalize_vote_label(datum["prediction"], datum.get("options", []))
            if label is None:
                # Conservative fallback: unknown label -> maximum uncertainty
                datum.setdefault("votes", [])
                datum["p_choice1"] = 0.5
                datum["p_choice2"] = 0.5
                datum["entropy"] = math.log(2.0)
                datum["weight"] = 0.0
                return

            votes = [label]
            c1 = 1 if label == "Choice 1" else 0
            c2 = 1 if label == "Choice 2" else 0
            p1, p2, ent, w = _compute_vote_stats(c1, c2)

            datum["votes"] = votes
            datum["p_choice1"] = p1
            datum["p_choice2"] = p2
            datum["entropy"] = ent
            datum["weight"] = w
            return

        # If no prediction exists, do nothing here.
        return

    def process_item(idx: int, datum: Dict[str, Any]) -> Tuple[int, bool]:
        # Si déjà prédit lors d'une exécution précédente, on passe
        if "prediction" in datum:
            _ensure_uncertainty_fields(datum)
            return idx, True

        prompt: str = datum["prepared"]

        # Self-consistency: collect M votes, then optionally re-query extra votes if uncertain.
        votes: List[str] = []
        choice1_count = 0
        choice2_count = 0

        def _one_completion() -> Tuple[Optional[str], Optional[str], Optional[str]]:
            completion, error = delayed_completion(
                client=client,
                prompt=prompt,
                delay_in_seconds=float(args.delay),
                max_trials=int(args.max_trials),
                model=args.ollama_model,
            )
            if completion is None:
                return None, None, str(error)
            content, results = post_process(completion, datum["options"])
            # Keep original output fields intact; results/content used below.
            return str(content) if content is not None else "", str(results) if results is not None else "", None

        # Initial sampling budget
        initial_M = max(1, int(args.num_responses))

        # Collect initial votes
        first_content: Optional[str] = None
        first_prediction: Optional[str] = None

        for _ in range(initial_M):
            content, results, err = _one_completion()
            if content is None or results is None:
                print(f"Error at index {idx}: {err}")
                return idx, False

            if first_content is None:
                first_content = content
            if first_prediction is None:
                first_prediction = results

            label = _normalize_vote_label(results, datum.get("options", []))
            if label is None:
                # If parsing fails, treat as failure (consistent with original strictness).
                print(f"Error at index {idx}: could not normalize prediction '{results}'")
                return idx, False

            votes.append(label)
            if label == "Choice 1":
                choice1_count += 1
            else:
                choice2_count += 1

        # Compute stats
        p1, p2, ent, w = _compute_vote_stats(choice1_count, choice2_count)
        max_p = max(p1, p2)

        # Active re-query (optional): only if configured (extra>0 and threshold in (0,1))
        requery_threshold = float(getattr(args, "requery_threshold", 0.0) or 0.0)
        requery_extra = int(getattr(args, "requery_extra", 0) or 0)

        if requery_extra > 0 and 0.0 < requery_threshold < 1.0 and max_p < requery_threshold:
            for _ in range(requery_extra):
                content, results, err = _one_completion()
                if content is None or results is None:
                    print(f"Error at index {idx}: {err}")
                    return idx, False

                label = _normalize_vote_label(results, datum.get("options", []))
                if label is None:
                    print(f"Error at index {idx}: could not normalize prediction '{results}'")
                    return idx, False

                votes.append(label)
                if label == "Choice 1":
                    choice1_count += 1
                else:
                    choice2_count += 1

            p1, p2, ent, w = _compute_vote_stats(choice1_count, choice2_count)

        # Preserve original output fields:
        # - content: keep first content (original behavior when M=1)
        # - prediction: keep first prediction (original behavior when M=1)
        datum["content"] = first_content if first_content is not None else ""
        datum["prediction"] = first_prediction if first_prediction is not None else ""

        # Add Contribution #3 fields
        datum["votes"] = votes
        datum["p_choice1"] = p1
        datum["p_choice2"] = p2
        datum["entropy"] = ent
        datum["weight"] = w

        return idx, True

    max_workers = args.num_threads  # Parallélisme (doit correspondre à OLLAMA_NUM_PARALLEL)
    save_counter = 0

    print(f"🚀 Lancement de l'inférence avec {max_workers} requêtes en parallèle...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prépare toutes les tâches
        futures = {executor.submit(process_item, i, d): i for i, d in enumerate(data)}

        # Traite les tâches au fur et à mesure qu'elles se terminent
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
            idx = futures[future]
            success = future.result()

            if not success:
                print(f"Sauvegarde des données après échec à l'index {idx}.")
                with open(pred_path, "w") as f:
                    json.dump(data, f, indent=2)
                raise RuntimeError(f"Ollama completion failed at idx={idx}")

            save_counter += 1
            if save_counter >= args.save_every:
                with open(pred_path, "w") as f:
                    json.dump(data, f, indent=2)
                save_counter = 0

    # Sauvegarde finale
    with open(pred_path, "w") as f:
        json.dump(data, f, indent=2)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Core arguments
    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--data_path", default=None, type=str, help="Specific path to input json (optional if input_dir is set)")

    # New arguments for DVC path handling
    parser.add_argument("--input_dir", default=None, type=str, help="Directory containing input json files")
    parser.add_argument("--output_dir", default=None, type=str, help="Directory to save output json files")

    # Model name alias (for compatibility with bash script)
    parser.add_argument("--model_name", default=None, type=str, help="Alias for ollama_model")

    # Inference settings
    parser.add_argument("--delay", type=float, default=0.0)  # Faster local inference
    parser.add_argument("--max_trials", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=50)

    # Self-consistency sampling
    parser.add_argument("--num_responses", type=int, default=1, help="Number of LLM responses per triplet (self-consistency M).")

    # Active re-query (optional; defaults disable to preserve original behavior)
    parser.add_argument(
        "--requery_threshold",
        type=float,
        default=0.0,
        help="If max(p_choice1, p_choice2) < threshold, query additional responses (0 disables).",
    )
    parser.add_argument(
        "--requery_extra",
        type=int,
        default=0,
        help="Number of additional responses to query when uncertain (0 disables).",
    )

    parser.add_argument("--num_threads", type=int, default=16, help="Parallélisme LLM")

    # Add Ollama config flags (host, model, temp, etc.)
    add_ollama_cli_args(parser)

    args = parser.parse_args()
    predict(args)