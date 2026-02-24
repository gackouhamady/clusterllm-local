# convert_triplet_self.py
import json, os
import h5py
import argparse
import random
import numpy as np
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=None, type=str)
parser.add_argument("--pred_path", default=None, type=str)   # triplets.json
parser.add_argument("--feat_path", default=None, type=str)   # .h5/.hdf5 with key 'embeds'
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--data_path", default=None, type=str)
args = parser.parse_args()

def get_text(ex):
    for k in ("text", "input", "sentence", "utterance", "query"):
        if k in ex and isinstance(ex[k], str):
            return ex[k]
    return str(ex)

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

    if choice1_dist <= choice2_dist:
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

output_name = os.path.basename(args.pred_path).replace(".json", "-self-train.json")
output_path = os.path.join(args.output_path, output_name)

os.makedirs(args.output_path, exist_ok=True)
with open(output_path, 'w', encoding="utf-8") as f:
    json.dump(out_data, f, ensure_ascii=False)

print(f"✅ Self-labeling réussi : {len(out_data)} triplets dans {output_path}")
print(f"ℹ️ skipped: {skipped}")
