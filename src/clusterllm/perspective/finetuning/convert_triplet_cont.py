# convert_triplet.py
import json, os
import argparse
import random
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=None, type=str)
parser.add_argument("--pred_path", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--data_path", default=None, type=str)
parser.add_argument("--e5", action="store_true",
                    help="E5 does not allow instructions.")
args = parser.parse_args()

def get_text(ex):
    """Dataset-agnostic: try common fields."""
    for k in ("text", "input", "sentence", "utterance", "query"):
        if k in ex and isinstance(ex[k], str):
            return ex[k]
    # last resort: stringify
    return str(ex)

def normalize_choice(pred_val):
    """
    Return 1 or 2 if we can parse a prediction, else None.
    Accepts:
      - [' 1'] / [' 2']
      - ['Choice 1'] / ['Choice 2']
      - ' 1' / ' 2' / '1' / '2'
      - 'Choice 1' / 'Choice 2'
      - sometimes LLM output like: "I choose Choice 1"
    """
    if pred_val is None:
        return None

    # pred can be list like ["Choice 1"]
    if isinstance(pred_val, list):
        if len(pred_val) != 1:
            return None
        pred_val = pred_val[0]

    # pred can be dict depending on some pipelines
    if isinstance(pred_val, dict):
        # common keys
        for k in ("content", "text", "prediction", "answer"):
            if k in pred_val:
                return normalize_choice(pred_val[k])
        return None

    if not isinstance(pred_val, str):
        pred_val = str(pred_val)

    s = pred_val.strip().lower()

    # canonical matches
    if s in {"1", " 1", "choice 1"}:
        return 1
    if s in {"2", " 2", "choice 2"}:
        return 2

    # fuzzy matches (LLM verbose)
    if "choice 1" in s or s.endswith("1"):
        return 1
    if "choice 2" in s or s.endswith("2"):
        return 2

    return None

# load files
with open(args.pred_path, 'r', encoding="utf-8") as f:
    pred_data = json.load(f)

with open(args.data_path, 'r', encoding="utf-8") as f:
    inp_data = [json.loads(l) for l in f if l.strip()]

# prompt logic identical spirit to paper, but robust fallback
if not args.e5:
    with open("prompts.json", 'r', encoding="utf-8") as f:
        prompts_dict = json.load(f)
    prompt = prompts_dict.get(args.dataset)
    if prompt is None:
        # fallback: try common alias
        prompt = prompts_dict.get("banking77")
    if prompt is None and len(prompts_dict) > 0:
        # fallback: first prompt available (keeps script usable for any dataset)
        prompt = next(iter(prompts_dict.values()))
else:
    prompt = "query: "

out_data = []
skipped = 0

for pd in pred_data:
    # require indices (paper logic)
    if not all(k in pd for k in ("query_idx", "choice1_idx", "choice2_idx")):
        skipped += 1
        continue

    pick = normalize_choice(pd.get("prediction"))
    if pick is None:
        skipped += 1
        continue

    q = pd["query_idx"]
    c1 = pd["choice1_idx"]
    c2 = pd["choice2_idx"]

    # bounds safety
    if not (0 <= q < len(inp_data) and 0 <= c1 < len(inp_data) and 0 <= c2 < len(inp_data)):
        skipped += 1
        continue

    if pick == 1:
        pos = get_text(inp_data[c1])
        neg = get_text(inp_data[c2])
    else:
        pos = get_text(inp_data[c2])
        neg = get_text(inp_data[c1])

    out_data.append({
        "query": [prompt, get_text(inp_data[q])],
        "pos": [prompt, pos],
        "neg": [prompt, neg],
        "task_name": args.dataset,
        "query_idx": q,
        "choice1_idx": c1,
        "choice2_idx": c2
    })


output_name = os.path.basename(args.pred_path).replace(".json", "_train.json")
output_path = os.path.join(args.output_path, output_name)

os.makedirs(args.output_path, exist_ok=True)
with open(output_path, 'w', encoding="utf-8") as f:
    json.dump(out_data, f, ensure_ascii=False)

print(f"✅ Conversion Standard réussie : {len(out_data)} triplets dans {output_path}")
print(f"ℹ️ skipped: {skipped}")
