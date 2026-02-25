import os
import json
import random
import h5py
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans


def load_data(args):
    return [json.loads(l) for l in open(args.data_path, 'r', encoding='utf-8')]


def load_feat(args):
    # Charge les embeddings initiaux extraits par Instructor
    with h5py.File(args.embed_path, 'r') as f:
        X = np.asarray(f['embeds'])
    return X


def entropy(vals):
    # Calcul de l'entropie sur les clusters les plus proches [cite: 137, 138]
    vals = np.asarray(vals)
    vals /= (vals.sum() + 1e-9)
    return - (vals * np.log(vals + 1e-9)).sum()


def _safe_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return v


def _load_llm_entropy_by_query(uncertainty_path: str):
    """
    Optional Contribution #3 input loader.

    Expected input: JSON list (or dict containing a list) of triplet prediction objects that MAY contain:
      - query_idx : int
      - entropy   : float

    Returns:
      dict: query_idx -> mean_entropy_over_triplets_for_that_query
    """
    if not uncertainty_path or not os.path.exists(uncertainty_path):
        return {}

    with open(uncertainty_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Accept list or dict-wrapped list
    if isinstance(obj, dict):
        # common fallbacks
        if "triplets" in obj and isinstance(obj["triplets"], list):
            items = obj["triplets"]
        elif "data" in obj and isinstance(obj["data"], list):
            items = obj["data"]
        elif "predictions" in obj and isinstance(obj["predictions"], list):
            items = obj["predictions"]
        else:
            # If dict is actually a single item, wrap it; otherwise empty
            items = [obj] if "query_idx" in obj else []
    elif isinstance(obj, list):
        items = obj
    else:
        items = []

    sums = {}
    counts = {}

    for it in items:
        if not isinstance(it, dict):
            continue
        if "query_idx" not in it:
            continue
        ent = _safe_float(it.get("entropy"))
        if ent is None:
            continue
        q = int(it["query_idx"])
        sums[q] = sums.get(q, 0.0) + float(ent)
        counts[q] = counts.get(q, 0) + 1

    out = {}
    for q, s in sums.items():
        c = counts.get(q, 0)
        if c > 0:
            out[q] = s / c
    return out


def generate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "triplets.json")

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_data(args)
    inp = [d['text'] for d in data]
    labels = [d['label'] for d in data]
    X = load_feat(args)

    X = StandardScaler().fit_transform(X)

    # Clustering initial pour définir l'espace de recherche [cite: 123]
    if args.scale == "small":
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=args.max_distance).fit(X)
    else:
        clustering = MiniBatchKMeans(n_clusters=100, random_state=args.seed).fit(X)

    preds = clustering.labels_
    n_clusters = len(set(preds))
    print(f"Estimated number of clusters: {n_clusters}")

    cluster_centers = []
    class_member_inds = {}
    for i in range(n_clusters):
        mask = (preds == i)
        cluster_centers.append(X[mask].mean(0))
        class_member_inds[i] = np.where(mask)[0]
    cluster_centers = np.stack(cluster_centers)

    # Identification des clusters les plus proches (K_closest) [cite: 132]
    num_closest = max(2, round(n_clusters * args.close_cluster_prop))
    options = []
    entropies = []
    for idx in range(len(X)):
        dist = ((X[idx] - cluster_centers) ** 2).sum(-1)
        prob = (1 + dist) ** (-1)  # soft assignment [cite: 128]
        prob /= prob.sum()
        sorted_prob = np.argsort(prob)[::-1][:num_closest]
        options.append(sorted_prob)
        entropies.append(entropy(prob[sorted_prob]))

    # -------------------------------
    # Contribution #3 (optional): uncertainty-aware acquisition
    # A(t) = alpha * Uemb + beta * entropy
    # - Uemb: embedding-derived uncertainty already computed as entropies[idx]
    # - entropy: LLM uncertainty loaded from uncertainty_path (per query_idx mean), if provided
    # Fallback: original behavior if uncertainty_path missing/empty
    # -------------------------------
    llm_entropy_by_query = _load_llm_entropy_by_query(getattr(args, "uncertainty_path", None))
    use_uncertainty = bool(llm_entropy_by_query)

    alpha = float(getattr(args, "acq_alpha", 1.0))
    beta = float(getattr(args, "acq_beta", 1.0))

    if use_uncertainty:
        acq_scores = []
        for idx in range(len(X)):
            u_emb = float(entropies[idx])
            u_llm = float(llm_entropy_by_query.get(idx, 0.0))
            acq_scores.append(alpha * u_emb + beta * u_llm)
        # Sélection des ancres à haut score d'acquisition (remplace tri sur entropies si uncertainty_path fourni)
        sorted_ent = np.argsort(acq_scores)[::-1][:int(len(X) * args.large_ent_prop)]
    else:
        # Sélection des ancres à haute entropie [cite: 120, 141] (ORIGINAL)
        sorted_ent = np.argsort(entropies)[::-1][:int(len(X) * args.large_ent_prop)]

    triplets = []
    while len(triplets) < args.num_queries:
        for idx in sorted_ent:
            cur_options = options[idx].tolist()
            if len(cur_options) < 2:
                continue
            cluster1, cluster2 = random.sample(cur_options, 2)
            choice1 = random.choice(class_member_inds[cluster1])
            choice2 = random.choice(class_member_inds[cluster2])
            if (idx, choice1, choice2) not in triplets and choice1 != idx and choice2 != idx:
                triplets.append((idx, choice1, choice2))
                if len(triplets) >= args.num_queries:
                    break

    result = []
    for trip in triplets:
        # Format triplet: <anchor, choice1, choice2> [cite: 109]
        result.append({
            "input": f"Query: {inp[trip[0]]}\nChoice 1: {inp[trip[1]]}\nChoice 2: {inp[trip[2]]}\nChoice",
            "task": args.dataset,
            "query_idx": int(trip[0]),
            "choice1_idx": int(trip[1]),
            "choice2_idx": int(trip[2]),
            # ✅ required by predict.py
            "options": ["Choice 1", "Choice 2"],
            # optional but harmless (keeps schema closer to the original)
            "output": None
        })

    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✅ {len(result)} triplets sauvegardés dans {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--embed_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scale", type=str, default="small")
    parser.add_argument("--num_queries", type=int, default=1024)
    parser.add_argument("--large_ent_prop", type=float, default=0.20)
    parser.add_argument("--close_cluster_prop", type=float, default=0.02)
    parser.add_argument("--max_distance", type=float, default=67)
    parser.add_argument("--seed", type=int, default=42)

    # Contribution #3 (optional): uncertainty-aware acquisition
    parser.add_argument(
        "--uncertainty_path",
        type=str,
        default=None,
        help="Optional JSON path with prior LLM uncertainties (expects per-item fields: query_idx, entropy).",
    )
    # Optional weights (kept optional; defaults are neutral)
    parser.add_argument("--acq_alpha", type=float, default=1.0, help="Weight for embedding uncertainty Uemb.")
    parser.add_argument("--acq_beta", type=float, default=1.0, help="Weight for LLM uncertainty entropy.")

    args = parser.parse_args()
    generate(args)