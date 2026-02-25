# convert_triplet_self.py
"""
convert_triplet_self_cont.py

CLUSTERLLM-style self-training conversion: build triplets using embedding distances (self-labeling).

STRICT RULE (respected):
- The original self-labeling logic (pos/neg assignment by distance) is NOT modified.
- Only EXTENDS the output schema to propagate soft supervision fields when available.

Contribution #3 extension (uncertainty-aware supervision)
--------------------------------------------------------
Input (optional new fields per triplet item from pred_path JSON):
  - p_choice1 : float in [0,1]  (empirical probability of choosing Choice 1)
  - weight    : float in [0,1]  (confidence weight, e.g., 1 - H/log(2))

Output (added fields per training example):
  - soft_pos_prob   : float in [0,1]  (probability that the chosen 'pos' is correct)
  - example_weight  : float in [0,1]  (training weight)

Mapping requirement:
- MUST preserve original pos/neg logic:
    if choice1_dist <= choice2_dist: pos = choice1, neg = choice2
    else:                            pos = choice2, neg = choice1
- soft_pos_prob MUST be consistent with the chosen pos:
    if pos == choice1 => soft_pos_prob = p_choice1
    if pos == choice2 => soft_pos_prob = 1 - p_choice1

Fallbacks (backward compatible):
- If p_choice1 or weight is missing/invalid:
    soft_pos_prob = 1.0
    example_weight = 1.0
"""

import json
import os
import h5py
import argparse
import random
import numpy as np
from typing import Any, Dict, Optional

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=None, type=str)
parser.add_argument("--pred_path", default=None, type=str)   # triplets.json
parser.add_argument("--feat_path", default=None, type=str)   # .h5/.hdf5 with key 'embeds'
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--data_path", default=None, type=str)
args = parser.parse_args()


def get_text(ex: Dict[str, Any]) -> str:
    for k in ("text", "input", "sentence", "utterance", "query"):
        if k in ex and isinstance(ex[k], str):
            return ex[k]
    return str(ex)


def _safe_float(x: Any) -> Optional[float]:
    """Parse float safely; return None if invalid."""
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return v


def _clip01(v: float) -> float:
    """Clip value to [0,1]."""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


with open(args.pred_path, 'r', encoding="utf-8") as f:
    triplets = json.load(f)

with open(args.data_path, 'r', encoding="utf-8") as f:
    inp_data = [json.loads(l) for l in f if l.strip()]

with open("prompts.json", 'r', encoding="utf-8") as f:
    prompts_dict = json.load(f)
prompt = prompts_dict.get(args.dataset) or prompts_dict.get("banking77") or next(iter(prompts_dict.values()))

# embeddings HDF5
with h5py.File(args.feat_path, 'r') as f:
    if "embeds" not in f:
        raise KeyError(f"❌ '{args.feat_path}' ne contient pas la clé 'embeds'. Clés dispo: {list(f.keys())}")
    embeds = np.asarray(f["embeds"])

out_data = []
skipped = 0

for pd in triplets:
    if not all(k in pd for k in ("query_idx", "choice1_idx", "choice2_idx")):
        skipped += 1
        continue

    q = int(pd["query_idx"])
    c1 = int(pd["choice1_idx"])
    c2 = int(pd["choice2_idx"])

    if not (0 <= q < len(inp_data) and 0 <= c1 < len(inp_data) and 0 <= c2 < len(inp_data)):
        skipped += 1
        continue
    if not (0 <= q < len(embeds) and 0 <= c1 < len(embeds) and 0 <= c2 < len(embeds)):
        skipped += 1
        continue

    choice1_dist = float(((embeds[q] - embeds[c1]) ** 2).sum())
    choice2_dist = float(((embeds[q] - embeds[c2]) ** 2).sum())

    # -----------------------------
    # ORIGINAL SELF-LABELING LOGIC (UNCHANGED)
    # -----------------------------
    if choice1_dist <= choice2_dist:
        pos = get_text(inp_data[c1])
        neg = get_text(inp_data[c2])
        pos_is_choice1 = True
    else:
        pos = get_text(inp_data[c2])
        neg = get_text(inp_data[c1])
        pos_is_choice1 = False

    # -----------------------------
    # Contribution #3 EXTENSION ONLY
    # -----------------------------
    # Propagate soft supervision if available on pd; otherwise fallback to 1.0/1.0.
    p_choice1 = _safe_float(pd.get("p_choice1"))
    weight = _safe_float(pd.get("weight"))

    if p_choice1 is None or weight is None:
        soft_pos_prob = 1.0
        example_weight = 1.0
    else:
        p_choice1 = _clip01(p_choice1)
        example_weight = _clip01(weight)
        soft_pos_prob = p_choice1 if pos_is_choice1 else (1.0 - p_choice1)
        soft_pos_prob = _clip01(soft_pos_prob)

    out_data.append({
        "query": [prompt, get_text(inp_data[q])],
        "pos": [prompt, pos],
        "neg": [prompt, neg],
        "task_name": args.dataset,
        "query_idx": q,
        "choice1_idx": c1,
        "choice2_idx": c2,
        # Added fields (backward compatible)
        "soft_pos_prob": soft_pos_prob,
        "example_weight": example_weight,
    })

output_name = os.path.basename(args.pred_path).replace(".json", "-self-train.json")
output_path = os.path.join(args.output_path, output_name)

os.makedirs(args.output_path, exist_ok=True)
with open(output_path, 'w', encoding="utf-8") as f:
    json.dump(out_data, f, ensure_ascii=False)

print(f"✅ Self-labeling réussi : {len(out_data)} triplets dans {output_path}")
print(f"ℹ️ skipped: {skipped}")