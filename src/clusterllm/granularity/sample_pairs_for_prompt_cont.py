#!/usr/bin/env python3
"""
sample_pairs_for_prompt_cont.py

Stage 7.1 prompt builder for pairwise granularity supervision.

Contribution #3 (Uncertainty- and noise-aware supervision) EXTENSION
-------------------------------------------------------------------
This script DOES NOT change the original prompt template logic or the core
pos/neg selection logic. It ONLY EXTENDS the pipeline to optionally use
self-consistency uncertainty signals (from predicted pairs) to choose cleaner
demonstration examples.

Optional input:
  --uncertainty_path : JSON produced by predict_pairs_cont.py (or compatible),
                       containing per-pair fields such as:
                         sent1_idx, sent2_idx, num_clusters, p_yes, entropy, weight

If provided and usable:
  - We enrich sampled pairs with p_yes/entropy/weight when indices match.
  - We prefer more confident demonstrations by sorting with entropy ascending
    (lower entropy = more confident), while preserving the original num_clusters
    ranking preference (pos: small K; neg: large K).
  - If fields are missing or file is absent, behavior is unchanged.

Backward compatibility:
  - If --uncertainty_path is not provided (default), behavior is identical.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple, Optional

dataset2lp = {
    # Intent / Domain (banking & assistants)
    "bank77": "intent",
    "clinc150": "intent",
    "clinc_intent": "intent",
    "clinc_domain": "domain",
    "massive_intent": "intent",
    "massive_domain": "domain",
    "mtop_intent": "intent",
    "mtop_domain": "domain",

    # Emotion / Topic / Domain
    "go_emotions": "emotion",
    "reddit": "topic",
    "stackex": "topic",
    "arxiv": "domain",

    # Few-shot / type discovery style
    "few_event": "event",
    "few_nerd_nat": "entity",
    "few_rel_nat": "relation",
}

def _norm_yesno(x: Any) -> str:
    s = str(x).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return "Yes"
    if s in ("no", "n", "false", "0"):
        return "No"
    return str(x).strip()

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def prepare_prompt(pos_pairs: List[Dict[str, Any]], neg_pairs: List[Dict[str, Any]], label_property: str) -> str:
    # Strict prompt formatting: demonstrations + final instruction
    # IMPORTANT: keep answers strictly "Yes" or "No" (no extra tokens).
    ex_pos = (
        "[Example<IDX>]\n"
        "Sentence 1: <SENT1>\n"
        "Sentence 2: <SENT2>\n"
        "Answer: Yes\n"
        "Reason: Both {lp}s are <LABEL>.\n\n"
    ).format(lp=label_property)

    ex_neg = (
        "[Example<IDX>]\n"
        "Sentence 1: <SENT1>\n"
        "Sentence 2: <SENT2>\n"
        "Answer: No\n"
        "Reason: Sentence 1 has {lp} <LABEL1> and Sentence 2 has {lp} <LABEL2>.\n\n"
    ).format(lp=label_property)

    instruction = (
        f"Task: Determine whether the {label_property}s of the two sentences below belong to the same "
        f"{label_property} category.\n"
        f"Rules:\n"
        f"- Answer with exactly one token: Yes or No.\n"
        f"- Do not add explanations.\n\n"
    )

    chunks = []
    for i, p in enumerate(pos_pairs, start=1):
        chunks.append(
            ex_pos.replace("<IDX>", str(i))
                 .replace("<SENT1>", str(p["sent1"]))
                 .replace("<SENT2>", str(p["sent2"]))
                 .replace("<LABEL>", str(p["label"]))
        )

    base = len(pos_pairs)
    for j, p in enumerate(neg_pairs, start=1):
        idx = base + j
        chunks.append(
            ex_neg.replace("<IDX>", str(idx))
                 .replace("<SENT1>", str(p["sent1"]))
                 .replace("<SENT2>", str(p["sent2"]))
                 .replace("<LABEL1>", str(p["label1"]))
                 .replace("<LABEL2>", str(p["label2"]))
        )

    chunks.append(instruction)
    return "".join(chunks)

def _extract_pairs(sampled_pairs_json: Any) -> List[Dict[str, Any]]:
    # Accept multiple schemas
    if isinstance(sampled_pairs_json, dict):
        for key in ("test_inputs", "pairs", "data", "inputs"):
            if key in sampled_pairs_json and isinstance(sampled_pairs_json[key], list):
                return sampled_pairs_json[key]
    if isinstance(sampled_pairs_json, list):
        return sampled_pairs_json
    raise ValueError("Unsupported sampled pairs schema (expected dict with test_inputs/pairs/... or list).")

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return v

def _clip01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v

def _load_uncertainty_map(path: Optional[str]) -> Dict[Tuple[int, int, Optional[int]], Dict[str, float]]:
    """
    Load a mapping for quick lookup:
      (sent1_idx, sent2_idx, num_clusters|None) -> {"p_yes":..., "entropy":..., "weight":...}
    Accepts predicted pairs outputs that are either:
      - dict with 'test_inputs' or 'pairs' list
      - list of dicts directly
    """
    if not path or not os.path.exists(path):
        return {}

    obj = _load_json(path)
    if isinstance(obj, dict):
        items = obj.get("test_inputs", obj.get("pairs", []))
        if not isinstance(items, list):
            items = [obj] if isinstance(obj, dict) else []
    elif isinstance(obj, list):
        items = obj
    else:
        items = []

    out: Dict[Tuple[int, int, Optional[int]], Dict[str, float]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        if "sent1_idx" not in it or "sent2_idx" not in it:
            continue

        i = int(it["sent1_idx"])
        j = int(it["sent2_idx"])
        a, b = (i, j) if i <= j else (j, i)
        step = it.get("num_clusters")
        step_i: Optional[int] = int(step) if step is not None else None

        p_yes = _safe_float(it.get("p_yes"))
        ent = _safe_float(it.get("entropy"))
        w = _safe_float(it.get("weight"))

        rec: Dict[str, float] = {}
        if p_yes is not None:
            rec["p_yes"] = _clip01(p_yes)
        if ent is not None:
            rec["entropy"] = float(ent)
        if w is not None:
            rec["weight"] = _clip01(w)

        if rec:
            out[(a, b, step_i)] = rec
            # also store pair-only fallback
            if (a, b, None) not in out:
                out[(a, b, None)] = rec

    return out

def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out_prompt_json = os.path.join(args.out_dir, "prompt.json")

    # Load prompt template (read-only). We do NOT overwrite it.
    base_prompts: Dict[str, Any] = {}
    if os.path.exists(args.prompt_path):
        try:
            base_prompts = _load_json(args.prompt_path) or {}
        except Exception:
            base_prompts = {}

    # Load sampled pairs and dataset jsonl
    all_pairs_raw = _extract_pairs(_load_json(args.sampled_pair_path))
    data = _load_jsonl(args.data_path)

    if len(data) == 0:
        raise ValueError(f"Empty dataset file: {args.data_path}")

    # Optional Contribution #3 uncertainty enrichment
    unc_map = _load_uncertainty_map(getattr(args, "uncertainty_path", None))
    use_unc = bool(unc_map)

    # Sample pairs
    n = min(args.num_sampled, len(all_pairs_raw))
    if n <= 0:
        raise ValueError("num_sampled must be > 0 and sampled_pair file must contain pairs.")

    sampled = random.sample(all_pairs_raw, n)

    # Normalize output field if present
    for p in sampled:
        if "output" in p:
            p["output"] = _norm_yesno(p["output"])
        elif "prediction" in p:
            p["output"] = _norm_yesno(p["prediction"])

        # Contribution #3: enrich with p_yes/entropy/weight if available
        if use_unc and "sent1_idx" in p and "sent2_idx" in p:
            i = int(p["sent1_idx"])
            j = int(p["sent2_idx"])
            a, b = (i, j) if i <= j else (j, i)
            step = p.get("num_clusters")
            step_i: Optional[int] = int(step) if step is not None else None
            rec = unc_map.get((a, b, step_i)) or unc_map.get((a, b, None))
            if rec:
                if "p_yes" in rec:
                    p["p_yes"] = rec["p_yes"]
                if "entropy" in rec:
                    p["entropy"] = rec["entropy"]
                if "weight" in rec:
                    p["weight"] = rec["weight"]
                # If no hard output, derive from soft p_yes
                if "output" not in p and "p_yes" in p:
                    p["output"] = "Yes" if float(p["p_yes"]) >= 0.5 else "No"

    # Split pos/neg with fallback if missing labels
    pos = [p for p in sampled if _norm_yesno(p.get("output", "")) == "Yes"]
    neg = [p for p in sampled if _norm_yesno(p.get("output", "")) == "No"]

    # If the file doesn't contain outputs, fallback using ground-truth labels if possible
    if (len(pos) == 0 and len(neg) == 0):
        tmp_pos, tmp_neg = [], []
        for p in sampled:
            i = int(p["sent1_idx"])
            j = int(p["sent2_idx"])
            same = str(data[i].get("label")) == str(data[j].get("label"))
            (tmp_pos if same else tmp_neg).append(p)
        pos, neg = tmp_pos, tmp_neg

    # Rank by num_clusters if available
    def nc(x: Dict[str, Any], default: float) -> float:
        try:
            return float(x.get("num_clusters", default))
        except Exception:
            return default

    # Contribution #3: prefer confident demo pairs when uncertainty is available.
    # We keep the original num_clusters preference and ONLY refine ordering by entropy (ascending).
    def ent(x: Dict[str, Any], default: float) -> float:
        try:
            v = _safe_float(x.get("entropy"))
            return float(v) if v is not None else default
        except Exception:
            return default

    # Choose best examples
    if use_unc:
        # pos: small K, then low entropy
        pos_sorted = sorted(pos, key=lambda x: (nc(x, 1e18), ent(x, 1e18)))[: args.num_for_prompt]
        # neg: large K, then low entropy
        neg_sorted = sorted(neg, key=lambda x: (-nc(x, -1e18), ent(x, 1e18)))[: args.num_for_prompt]
    else:
        pos_sorted = sorted(pos, key=lambda x: nc(x, 1e18))[: args.num_for_prompt]
        neg_sorted = sorted(neg, key=lambda x: nc(x, -1e18), reverse=True)[: args.num_for_prompt]

    if len(pos_sorted) == 0 or len(neg_sorted) == 0:
        raise ValueError(
            f"Not enough pos/neg pairs for prompt. Got pos={len(pos_sorted)}, neg={len(neg_sorted)}. "
            f"Increase --num_sampled or check your sampled pairs file."
        )

    # Enrich with sentences and labels
    def get_text(i: int) -> str:
        return str(data[i].get("text", data[i].get("sentence", "")))

    def get_label(i: int) -> str:
        return str(data[i].get("label", ""))

    for p in pos_sorted:
        i = int(p["sent1_idx"]); j = int(p["sent2_idx"])
        p["sent1"] = get_text(i)
        p["sent2"] = get_text(j)
        p["label"] = get_label(i)

    for p in neg_sorted:
        i = int(p["sent1_idx"]); j = int(p["sent2_idx"])
        p["sent1"] = get_text(i)
        p["sent2"] = get_text(j)
        p["label1"] = get_label(i)
        p["label2"] = get_label(j)

    label_property = dataset2lp.get(args.dataset, "category")
    formatted_prompt = prepare_prompt(pos_sorted, neg_sorted, label_property)

    # Write prompt file used by predict_pairs.py
    # Keep mapping schema: {"dataset_name": "prompt string"}
    out_obj = dict(base_prompts) if isinstance(base_prompts, dict) else {}
    out_obj[args.dataset] = formatted_prompt

    with open(out_prompt_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    # Optional: also dump readable txt
    with open(os.path.join(args.out_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(formatted_prompt)

    print(f"✅ Prompt generated for dataset='{args.dataset}'")
    print(f"   - JSON: {out_prompt_json}")
    print(f"   - TXT : {os.path.join(args.out_dir, 'prompt.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str, required=True, help="Read-only prompt template (JSON mapping).")
    parser.add_argument("--sampled_pair_path", type=str, required=True, help="Sampled pairs JSON produced by sample_pairs.py")
    parser.add_argument("--data_path", type=str, required=True, help="Dataset jsonl path")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_sampled", type=int, default=200)
    parser.add_argument("--num_for_prompt", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, required=True)

    # Contribution #3 (optional): uncertainty-aware demo selection
    parser.add_argument(
        "--uncertainty_path",
        type=str,
        default=None,
        help="Optional predicted pairs JSON (with p_yes/entropy/weight) to pick cleaner prompt demonstrations.",
    )

    args = parser.parse_args()
    main(args)