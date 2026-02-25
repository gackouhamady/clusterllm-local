import os
import json
import argparse
import numpy as np
from sklearn.metrics import fbeta_score


def load_data(args):
    data_path = args.data_path
    # Charge le fichier JSONL d'origine (test.jsonl)
    return [json.loads(l) for l in open(data_path, 'r')]


def _safe_float(x):
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


def _get_p_yes(pred: dict) -> float:
    """
    Contribution #3 (soft constraints):
    Return p_yes in [0,1] if available, else fall back to hard yes/no parsing.

    Backward compatible behavior:
      - If p_yes is missing, we parse 'prediction' or 'output' and return 1.0 for Yes, 0.0 for No.
    """
    p_yes = pred.get("p_yes", None)
    p_yes_f = _safe_float(p_yes)
    if p_yes_f is not None:
        return _clip01(p_yes_f)

    # Fallback to original hard logic
    p = pred.get("prediction", pred.get("output", ['No']))
    is_yes = False
    if isinstance(p, list) and len(p) > 0:
        is_yes = str(p[0]).lower() == 'yes'
    elif isinstance(p, str):
        is_yes = p.lower() == 'yes'
    return 1.0 if is_yes else 0.0


def _expected_fbeta_score(y_true_prob, y_pred_bin, beta: float = 0.92) -> float:
    """
    Expected F-beta when y_true is Bernoulli(p_i) and y_pred is deterministic in {0,1}.

    We interpret the CLUSTERLLM scoring as:
      y_true := LLM constraint ("Yes"=1, "No"=0)
      y_pred := clustering assignment match (same cluster=1, different=0)

    Soft extension:
      y_true_prob := p_yes (probability that the constraint is positive/must-link)
      y_pred_bin  := {0,1} from clustering assignment

    Expected counts:
      TP = sum p_i * y_pred
      FP = sum (1-p_i) * y_pred
      FN = sum p_i * (1-y_pred)

    Then compute:
      Precision = TP/(TP+FP)
      Recall    = TP/(TP+FN)
      F_beta    = (1+beta^2)*P*R / (beta^2*P + R)

    Returns 0.0 if denominators are 0.
    """
    p = np.asarray(y_true_prob, dtype=np.float64)
    y = np.asarray(y_pred_bin, dtype=np.float64)

    tp = float(np.sum(p * y))
    fp = float(np.sum((1.0 - p) * y))
    fn = float(np.sum(p * (1.0 - y)))

    prec_den = tp + fp
    rec_den = tp + fn

    precision = (tp / prec_den) if prec_den > 0.0 else 0.0
    recall = (tp / rec_den) if rec_den > 0.0 else 0.0

    beta2 = beta * beta
    den = beta2 * precision + recall
    if den <= 0.0:
        return 0.0
    return (1.0 + beta2) * precision * recall / den


def predict(args):
    # 1. Chargement des résultats de clustering (Hiérarchie)
    print(f"Loading hierarchy from: {args.clustering_results}")
    with open(args.clustering_results, 'r') as f:
        clustering_results = json.load(f)

    # Support flexible pour les noms de clés (clusters vs cluster)
    clustering = clustering_results.get('clusters', clustering_results.get('cluster'))
    nodes = clustering_results.get('nodes')
    children = clustering_results.get('children')

    if clustering is None or nodes is None or children is None:
        raise KeyError("Le fichier de clustering doit contenir les clés 'clusters' (ou 'cluster'), 'nodes' et 'children'.")

    # 2. Chargement des prédictions LLM (Paires Yes/No)
    print(f"Loading LLM predictions from: {args.pred_path}")
    with open(args.pred_path, 'r') as f:
        preds_data = json.load(f)

    # Extraction des inputs selon le format du fichier de prédiction
    if isinstance(preds_data, dict):
        # On cherche 'test_inputs' ou 'pairs' ou on prend le dict tel quel
        preds = preds_data.get('test_inputs', preds_data.get('pairs', []))
        if not preds and not any(k in preds_data for k in ['clusters', 'nodes']):
            preds = [preds_data]  # Cas d'un seul objet
    else:
        preds = preds_data

    # Définition de la plage de recherche pour K
    if "num_clusters" in preds_data and isinstance(preds_data["num_clusters"], list):
        num_clusters_range = preds_data["num_clusters"]
    else:
        num_clusters_range = list(range(args.min_clusters, args.max_clusters + 1))

    # Correction spécifique pour le mode 'large'
    if args.scale == "large":
        clustering[0] = list(range(0, args.max_clusters))

    # 3. Pré-calcul des assignations de clusters pour chaque K
    print("Computing cluster assignments for search range...")
    assign = {}
    for cls_idx in num_clusters_range:
        try:
            clst = clustering[-cls_idx]
            cur_assign = {}
            for nidx, node in enumerate(clst):
                for idx in nodes[str(node)]:
                    cur_assign[idx] = nidx
            assign[cls_idx] = cur_assign
        except IndexError:
            continue

    # 4. Calcul du score de consistance (F-beta)
    print(f"Calculating consistency scores (Beta=0.92) for K in {num_clusters_range[0]}..{num_clusters_range[-1]}")
    acc = []
    valid_range = []

    for cls_idx in num_clusters_range:
        if cls_idx not in assign:
            continue

        cur_true_prob = []   # p_yes from LLM (soft) or 0/1 (hard fallback)
        cur_assign_pair = []  # 0/1 from clustering (same cluster)

        cur_assign = assign[cls_idx]

        for pred in preds:
            # On vérifie que les indices existent dans l'assignation courante
            if pred['sent1_idx'] not in cur_assign or pred['sent2_idx'] not in cur_assign:
                continue

            sent1_assign = cur_assign[pred['sent1_idx']]
            sent2_assign = cur_assign[pred['sent2_idx']]

            # On ne calcule le score que si le niveau de cluster est cohérent
            if pred['num_clusters'] + 1 in num_clusters_range:
                # Contribution #3: soft constraints via expected satisfaction
                p_yes = _get_p_yes(pred)  # in [0,1]
                cur_true_prob.append(p_yes)
                cur_assign_pair.append(1 if sent1_assign == sent2_assign else 0)

        if cur_true_prob:
            # Use expected F-beta when p_yes is present; still works for hard 0/1.
            score = _expected_fbeta_score(cur_true_prob, cur_assign_pair, beta=0.92)
            acc.append(score)
            valid_range.append(cls_idx)

    # 5. Résultat Final
    if not acc:
        print("❌ Erreur: Aucune paire valide n'a pu être traitée.")
        return

    best_idx = np.argsort(acc)[::-1][0]
    final_k = valid_range[best_idx]

    print("\n" + "=" * 30)
    print(f"DATASET: {args.dataset}")

    # Tentative d'affichage du K réel si labels présents
    try:
        data_raw = load_data(args)
        if 'label' in data_raw[0]:
            labels = [l['label'] for l in data_raw]
            print(f"REAL K: {len(set(labels))}")
    except Exception:
        pass

    print(f"ESTIMATED K: {final_k}")
    top_10 = [valid_range[i] for i in np.argsort(acc)[::-1][:10]]
    print(f"TOP 10 CANDIDATES: {top_10}")
    print("=" * 30)

    # Sauvegarde du résultat
    res_path = os.path.join(os.path.dirname(args.pred_path), "final_granularity_results.json")
    with open(res_path, 'w') as f:
        json.dump({"n_clusters_pred": int(final_k), "top_candidates": top_10}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--clustering_results", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--scale", type=str, default="small")
    parser.add_argument("--min_clusters", type=int, default=2)
    parser.add_argument("--max_clusters", type=int, default=200)
    args = parser.parse_args()

    predict(args)