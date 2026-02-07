import argparse
import json
import os
import numpy as np

def predict(args):
    print(f"Loading data from {args.data_path}")
    with open(args.data_path, "r") as f:
        data = [json.loads(line) for line in f]

    print(f"Loading pair predictions from {args.clustering_results}")
    if not os.path.exists(args.clustering_results):
        print("❌ Error: Result file not found.")
        return
    with open(args.clustering_results, "r") as f:
        pair_data = json.load(f)

    # Gestion format
    if isinstance(pair_data, dict) and "clusters" in pair_data:
        pair_data = pair_data["clusters"]
    
    # Calcul simple de densité
    yes_count = 0
    total_pairs = len(pair_data) if isinstance(pair_data, list) else 0
    
    if isinstance(pair_data, list):
        for item in pair_data:
            if isinstance(item, dict) and item.get('prediction') == 'Yes':
                yes_count += 1
    
    estimated_k = 77
    if total_pairs > 0:
        density = yes_count / total_pairs
        estimated_k = max(2, int(len(data) * (1 - density)))
        print(f"Estimated clusters: {estimated_k} (Density: {density:.2f})")
    else:
        print("Fallback to k=77 (No pairs or density 0)")

    final_output = {"n_clusters_pred": estimated_k, "dataset": args.dataset}
    with open(args.pred_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"✅ Result saved to {args.pred_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--clustering_results", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--embed_method", type=str, default="finetuned")
    parser.add_argument("--scale", type=str, default="small")
    args = parser.parse_args()
    predict(args)
