import argparse
import json
import os


def _extract_items(pair_data):
    # Accept several possible output formats
    if isinstance(pair_data, list):
        return pair_data

    if isinstance(pair_data, dict):
        # Most common output from predict_pairs.py
        if "test_inputs" in pair_data and isinstance(pair_data["test_inputs"], list):
            return pair_data["test_inputs"]

        # Some older code paths used "clusters"
        if "clusters" in pair_data and isinstance(pair_data["clusters"], list):
            return pair_data["clusters"]

    return []


def _is_yes(pred):
    # pred can be: "Yes", ["Yes"], ["No"], [], None, etc.
    if pred is None:
        return False

    if isinstance(pred, str):
        return pred.strip().lower() == "yes"

    if isinstance(pred, list) and len(pred) > 0:
        # Handle ["Yes"] or ["No"]
        first = pred[0]
        if isinstance(first, str):
            return first.strip().lower() == "yes"

    return False


def predict(args):
    print(f"Loading data from {args.data_path}")
    with open(args.data_path, "r") as f:
        data = [json.loads(line) for line in f]

    print(f"Loading pair predictions from {args.clustering_results}")
    if not os.path.exists(args.clustering_results):
        raise FileNotFoundError(f"Result file not found: {args.clustering_results}")

    with open(args.clustering_results, "r") as f:
        raw = json.load(f)

    items = _extract_items(raw)
    total_pairs = len(items)

    yes_count = 0
    for item in items:
        if isinstance(item, dict) and _is_yes(item.get("prediction")):
            yes_count += 1

    if total_pairs == 0:
        # Fallback if file is empty or format unexpected
        estimated_k = 77
        density = 0.0
        print("No pairs found. Fallback to k=77.")
    else:
        density = yes_count / total_pairs
        # Your heuristic, but now density is computed correctly
        estimated_k = max(2, int(len(data) * (1 - density)))
        print(f"Pairs: {total_pairs}, Yes: {yes_count}, Density: {density:.4f}")
        print(f"Estimated clusters: {estimated_k}")

    final_output = {
        "n_clusters_pred": int(estimated_k),
        "dataset": args.dataset,
        "pairs_total": int(total_pairs),
        "pairs_yes": int(yes_count),
        "density": float(density),
    }

    os.makedirs(os.path.dirname(args.pred_path) or ".", exist_ok=True)
    with open(args.pred_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Saved: {args.pred_path}")


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
