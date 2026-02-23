#!/usr/bin/env python3
import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

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

    # Choose best examples
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
    args = parser.parse_args()
    main(args)