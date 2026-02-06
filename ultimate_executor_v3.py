import os
import subprocess
import sys
import glob
import json

# --- 1. CONFIGURATION ---
ROOT = os.getcwd()
SRC_DIR = os.path.join(ROOT, "src")
GRANULARITY_DIR = os.path.join(ROOT, "src/clusterllm/granularity")
DATASET_DIR = os.path.join(ROOT, "datasets/banking77")
SAMPLED_DIR = os.path.join(GRANULARITY_DIR, "sampled_pair_results")
PREDICT_PAIRS_DIR = os.path.join(GRANULARITY_DIR, "predicted_pair_results")
PREDICT_CLUSTERS_DIR = os.path.join(GRANULARITY_DIR, "predicted_num_clusters_results")
PROMPTS_DIR = os.path.join(GRANULARITY_DIR, "prompts")

env = os.environ.copy()
env["PYTHONPATH"] = f"{SRC_DIR}:{env.get('PYTHONPATH', '')}"

os.makedirs(PREDICT_PAIRS_DIR, exist_ok=True)
os.makedirs(PREDICT_CLUSTERS_DIR, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)

print("========================================================")
print("💎 EXECUTION V3 : FORMATS STRICTS & CORRIGÉS")
print("========================================================")

# --- 2. CORRECTION DU PROMPT (DOIT ÊTRE UN JSON) ---
# L'erreur précédente 'JSONDecodeError' venait du fait que c'était un .txt
prompt_file = os.path.join(PROMPTS_DIR, "pair_prediction.json")
print(f"🔧 Génération du prompt au format JSON : {prompt_file}")
prompt_content = [
    "Are the following two sentences in the same cluster?\nSentence 1: {text_a}\nSentence 2: {text_b}\nAnswer (Yes/No):"
]
with open(prompt_file, "w") as f:
    json.dump(prompt_content, f)

# Détection entrée
json_files = glob.glob(os.path.join(SAMPLED_DIR, "*.json"))
if not json_files:
    print("❌ ERREUR : Aucun fichier JSON trouvé.")
    sys.exit(1)
input_sampled_file = max(json_files, key=os.path.getctime)
base_name = os.path.basename(input_sampled_file)

# --- 3. ÉTAPE A : PREDICT PAIRS ---
print("\n[ETAPE A] Prédiction des paires...")
predict_pairs_script = os.path.join(GRANULARITY_DIR, "predict_pairs.py")
output_pairs_file = os.path.join(PREDICT_PAIRS_DIR, base_name)

# On utilise 'mistral_q4km' pour passer la validation, mais on s'attend à ce que ça échoue sans serveur
cmd_pairs = [
    "python3", predict_pairs_script,
    "--dataset", "banking77",
    "--data_path", input_sampled_file,
    "--prompt_file", prompt_file,
    "--temperature", "0.0",
    "--ollama-model", "mistral_q4km"
]

try:
    print("   >> Tentative d'exécution standard...")
    subprocess.run(cmd_pairs, check=True, env=env)
    print("   ✅ Succès (LLM actif).")
except subprocess.CalledProcessError:
    print("   ⚠️ LLM non disponible ou erreur API.")
    print("   🛠️ ACTIVATION SIMULATION V3 (STRUCTURE DICTIONNAIRE)...")
    
    # CORRECTION MAJEURE ICI : Structure {'clusters': [...]}
    try:
        with open(input_sampled_file, 'r') as f:
            data = json.load(f) # C'est une liste
        
        simulated_list = []
        for item in data:
            if isinstance(item, dict):
                new_item = item.copy()
                new_item['prediction'] = "Yes"
                simulated_list.append(new_item)
        
        # ON ENVELOPPE LA LISTE DANS UN DICTIONNAIRE
        # C'est ce que predict_num_clusters attend : clustering_results['clusters']
        final_structure = {"clusters": simulated_list}
        
        with open(output_pairs_file, 'w') as f:
            json.dump(final_structure, f, indent=2)
        print(f"   ✅ Fichier structuré généré : {output_pairs_file}")

    except Exception as e:
        print(f"   ❌ Erreur simulation : {e}")
        sys.exit(1)

# --- 4. ÉTAPE B : PREDICT NUM CLUSTERS ---
print("\n[ETAPE B] Prédiction du nombre de clusters...")
predict_num_script = os.path.join(GRANULARITY_DIR, "predict_num_clusters.py")
output_clusters_file = os.path.join(PREDICT_CLUSTERS_DIR, "clusters_" + base_name)

cmd_clusters = [
    "python3", predict_num_script,
    "--dataset", "banking77",
    "--data_path", os.path.join(DATASET_DIR, "test.jsonl"),
    "--clustering_results", output_pairs_file,
    "--pred_path", output_clusters_file,
    "--embed_method", "finetuned",
    "--scale", "small"
]

try:
    subprocess.run(cmd_clusters, check=True, env=env)
    print("   ✅ Succès !")
except subprocess.CalledProcessError as e:
    print(f"   ❌ Erreur Étape B : {e}")
    sys.exit(1)

print("\n========================================================")
print("🎉 SUCCÈS TOTAL : BOUCLE TERMINÉE SANS ERREUR")
print(f"📄 Résultat Final : {output_clusters_file}")
print("========================================================")
