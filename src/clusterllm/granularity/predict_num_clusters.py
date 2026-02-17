import os
import json
import argparse
import numpy as np
from sklearn.metrics import fbeta_score

def load_data(args):
    data_path = args.data_path
    # Charge le fichier JSONL d'origine (test.jsonl)
    return [json.loads(l) for l in open(data_path, 'r')]

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
            preds = [preds_data] # Cas d'un seul objet
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
        if cls_idx not in assign: continue
        
        cur_preds = []
        cur_assign_pair = []
        cur_assign = assign[cls_idx]
        
        for pred in preds:
            # On vérifie que les indices existent dans l'assignation courante
            if pred['sent1_idx'] not in cur_assign or pred['sent2_idx'] not in cur_assign:
                continue
                
            sent1_assign = cur_assign[pred['sent1_idx']]
            sent2_assign = cur_assign[pred['sent2_idx']]
            
            # Extraction de la prédiction (supporte "Yes" ou ["Yes"])
            p = pred.get('prediction', pred.get('output', ['No']))
            is_yes = False
            if isinstance(p, list) and len(p) > 0:
                is_yes = str(p[0]).lower() == 'yes'
            elif isinstance(p, str):
                is_yes = p.lower() == 'yes'

            # On ne calcule le score que si le niveau de cluster est cohérent
            if pred['num_clusters'] + 1 in num_clusters_range:
                cur_preds.append(1 if is_yes else 0)
                cur_assign_pair.append(1 if sent1_assign == sent2_assign else 0)
        
        if cur_preds:
            score = fbeta_score(cur_preds, cur_assign_pair, pos_label=1, beta=0.92, zero_division=0)
            acc.append(score)
            valid_range.append(cls_idx)
    
    # 5. Résultat Final
    if not acc:
        print("❌ Erreur: Aucune paire valide n'a pu être traitée.")
        return

    best_idx = np.argsort(acc)[::-1][0]
    final_k = valid_range[best_idx]
    
    print("\n" + "="*30)
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
    print("="*30)

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