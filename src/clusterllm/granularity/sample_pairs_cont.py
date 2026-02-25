import pickle
# instead of looking at just the closest
# we look at pairs in multiple granularities
import os
import json
import h5py
import argparse
import random
import numpy as np
from copy import deepcopy

from sklearn.cluster import AgglomerativeClustering


def load_data(args):
    # data_path = f"../datasets/{args.dataset}/{args.scale}.jsonl"
    return [json.loads(l) for l in open(args.data_path, 'r')]


def load_feat(args):
    feat_path = args.feat_path

    # PATCH HYBRIDE (PKL + HDF5)
    if feat_path.endswith('.pkl'):
        with open(feat_path, 'rb') as f:
            return pickle.load(f)
    # Fallback HDF5 d'origine
    with h5py.File(feat_path, 'r') as f:
        X = f['embeds']
        X = np.asarray(X)
    return X


def _safe_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return v


def _load_llm_entropy_by_pair(uncertainty_path: str):
    """
    Optional Contribution #3 input loader for uncertainty-aware acquisition.

    Expected input: JSON file from a previous pair-prediction run that MAY contain:
      - list of dicts (direct)
      - dict with key 'test_inputs' or 'pairs' containing list

    Each item MAY contain:
      - sent1_idx : int
      - sent2_idx : int
      - num_clusters : int
      - entropy : float

    Returns:
      - ent_by_pair_step: dict[(i,j,step)] -> entropy
      - ent_by_pair:      dict[(i,j)] -> mean_entropy (fallback if step missing)
    """
    if not uncertainty_path or not os.path.exists(uncertainty_path):
        return {}, {}

    with open(uncertainty_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        items = obj.get("test_inputs", obj.get("pairs", []))
        if not isinstance(items, list):
            items = [obj] if all(k in obj for k in ("sent1_idx", "sent2_idx")) else []
    elif isinstance(obj, list):
        items = obj
    else:
        items = []

    ent_by_pair_step = {}
    sums = {}
    counts = {}

    for it in items:
        if not isinstance(it, dict):
            continue
        if "sent1_idx" not in it or "sent2_idx" not in it:
            continue

        ent = _safe_float(it.get("entropy"))
        if ent is None:
            continue

        i = int(it["sent1_idx"])
        j = int(it["sent2_idx"])
        a, b = (i, j) if i <= j else (j, i)

        # step-aware (preferred)
        if "num_clusters" in it:
            step = int(it["num_clusters"])
            ent_by_pair_step[(a, b, step)] = float(ent)

        # pair-only aggregation fallback
        sums[(a, b)] = sums.get((a, b), 0.0) + float(ent)
        counts[(a, b)] = counts.get((a, b), 0) + 1

    ent_by_pair = {}
    for k, s in sums.items():
        c = counts.get(k, 0)
        if c > 0:
            ent_by_pair[k] = s / c

    return ent_by_pair_step, ent_by_pair


def generate(args):
    os.makedirs(args.out_dir, exist_ok=True)
    # save_path = f"{args.out_dir}/{args.dataset}_embed={args.embed_method}_n={args.num_data}_m={args.max_query}_multigran{args.min_clusters}-{args.max_clusters}_seed={args.seed}.json"
    save_path = f"{args.out_dir}/{args.dataset}_embed={args.embed_method}_s={args.scale}_k={args.k}_multigran{args.min_clusters}-{args.max_clusters}_seed={args.seed}.json"
    print(save_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = load_data(args)
    inp = [d['text'] for d in data]
    # for analyzing purpose only
    labels = [d['label'] for d in data]
    X = load_feat(args)

    clustering = AgglomerativeClustering().fit(X)
    children = clustering.children_

    nodes = {idx: [idx] for idx in range(len(data))}
    cnt = len(data)
    clusters = []
    cur_clusters = list(range(len(data)))
    for child in children:
        nodes[cnt] = nodes[child[0]] + nodes[child[1]]

        cur_clusters.remove(child[0])
        cur_clusters.remove(child[1])
        cur_clusters.append(cnt)

        clusters.append(deepcopy(cur_clusters))
        cnt += 1

    # rerank examples according to distance to cluster center
    for idx in nodes:
        if len(nodes[idx]) <= 2:
            continue
        cur_embeds = X[nodes[idx]]
        cur_center = np.mean(cur_embeds, axis=0)
        cur_dists = ((cur_embeds - cur_center[None, :]) ** 2).sum(axis=-1)
        nodes[idx] = [nodes[idx][i] for i in np.argsort(cur_dists)]

    # -------------------------------
    # Contribution #3 (optional): uncertainty-aware acquisition
    # A(pair, step) = alpha * Uemb + beta * entropy
    # - Uemb: embedding ambiguity, defined here as similarity proxy 1/(1+||x_i-x_j||^2)
    # - entropy: LLM uncertainty loaded from uncertainty_path if provided
    # Fallback: original behavior if uncertainty_path missing/empty
    # -------------------------------
    ent_by_pair_step, ent_by_pair = _load_llm_entropy_by_pair(getattr(args, "uncertainty_path", None))
    use_uncertainty = bool(ent_by_pair_step) or bool(ent_by_pair)

    alpha = float(getattr(args, "acq_alpha", 1.0))
    beta = float(getattr(args, "acq_beta", 1.0))

    # small fixed candidate pool (kept internal to avoid changing original CLI contract)
    pool = 16
    pool_pairs = 32

    def _uemb(i: int, j: int) -> float:
        d2 = float(((X[i] - X[j]) ** 2).sum())
        return 1.0 / (1.0 + d2)

    def _llm_ent(i: int, j: int, step: int) -> float:
        a, b = (i, j) if i <= j else (j, i)
        v = ent_by_pair_step.get((a, b, step))
        if v is not None:
            return float(v)
        v = ent_by_pair.get((a, b))
        return float(v) if v is not None else 0.0

    all_test_pairs = []
    # while len(all_test_pairs) < args.max_query:
    for _ in range(args.k):
        # order = list(range(args.min_clusters, args.max_clusters))
        # random.shuffle(order)
        # for step in order:
        for step in range(args.min_clusters, args.max_clusters):
            cls_idx1, cls_idx2 = children[-step]  # look for the child before current step
            cls1 = nodes[cls_idx1]
            cls2 = nodes[cls_idx2]

            if not use_uncertainty:
                idx1 = random.choice(cls1)
                idx2 = random.choice(cls2)
            else:
                # Candidate selection: sample a small pool from each cluster (mix of near-center and random)
                c1 = cls1[:min(pool, len(cls1))]
                c2 = cls2[:min(pool, len(cls2))]

                # add random diversity
                if len(cls1) > len(c1):
                    c1 = c1 + random.sample(cls1, k=min(pool, len(cls1)))
                if len(cls2) > len(c2):
                    c2 = c2 + random.sample(cls2, k=min(pool, len(cls2)))

                # choose best pair according to acquisition
                best = None
                best_score = None
                for _t in range(pool_pairs):
                    i = random.choice(c1)
                    j = random.choice(c2)
                    if i == j:
                        continue
                    score = alpha * _uemb(i, j) + beta * _llm_ent(i, j, step)
                    if best_score is None or score > best_score:
                        best_score = score
                        best = (i, j)

                if best is None:
                    idx1 = random.choice(cls1)
                    idx2 = random.choice(cls2)
                else:
                    idx1, idx2 = best

            # p = (idx1, idx2)
            p = tuple(sorted([idx1, idx2]))
            if p not in [pp for (pp, _) in all_test_pairs]:
                all_test_pairs.append((p, step))
                # if len(all_test_pairs) >= args.max_query:
                #     break

    test_inputs = []
    for pair in all_test_pairs:
        ((p1, p2), step) = pair
        if random.random() > 0.5:
            input_txt = "Sentence 1: " + inp[p1] + "\nSentence 2: " + inp[p2]
            # for analyzing purpose
            if labels[p1] == labels[p2]:
                output = "Yes"
            else:
                output = "No"
            test_inputs.append({
                "input": input_txt,
                "output": output,
                "options": ['Yes', 'No'],
                "task": args.dataset,
                "sent1_idx": int(p1),
                "sent2_idx": int(p2),
                "num_clusters": int(step)
            })
        else:
            input_txt = "Sentence 1: " + inp[p1] + "\nSentence 2: " + inp[p2]
            # for analyzing purpose
            if labels[p1] == labels[p2]:
                output = "Yes"
            else:
                output = "No"
            test_inputs.append({
                "input": input_txt,
                "output": output,
                "options": ['Yes', 'No'],
                "task": args.dataset,
                "sent1_idx": int(p1),
                "sent2_idx": int(p2),
                "num_clusters": int(step)
            })

    with open(save_path, 'w') as f:
        json.dump({
            "test_inputs": test_inputs,
            "clusters": clusters,
            "nodes": nodes,
            "children": children.tolist()
        }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--embed_method", type=str, default='instructor')
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--feat_path", type=str, required=True)
    # parser.add_argument("--num_data", type=int, default=1024)
    parser.add_argument("--scale", default="small", type=str)
    parser.add_argument("--k", type=int, default=1,
                        help="# of times to sample from a pair of clusters")
    parser.add_argument("--out_dir", default="links", type=str)
    parser.add_argument("--min_clusters", default=2, type=int)
    parser.add_argument("--max_clusters", default=200, type=int)
    parser.add_argument("--seed", type=int, default=100)

    # Contribution #3 (optional): uncertainty-aware acquisition
    parser.add_argument(
        "--uncertainty_path",
        type=str,
        default=None,
        help="Optional JSON path with prior LLM uncertainties (expects per-item fields: sent1_idx, sent2_idx, (optional) num_clusters, entropy).",
    )
    parser.add_argument("--acq_alpha", type=float, default=1.0, help="Weight for embedding ambiguity Uemb.")
    parser.add_argument("--acq_beta", type=float, default=1.0, help="Weight for LLM uncertainty entropy.")

    args = parser.parse_args()
    generate(args)