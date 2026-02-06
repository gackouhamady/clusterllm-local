import os
import json
import subprocess
import sys

# --- CONFIGURATION ---
ROOT = os.getcwd()
CHECKPOINT_ROOT = os.path.join(ROOT, "src/clusterllm/perspective/finetuning/checkpoints")
FINETUNE_SCRIPT = os.path.join(ROOT, "src/clusterllm/perspective/finetuning/scripts/finetune.sh")
EMBED_SCRIPT_PATH = os.path.join(ROOT, "src/clusterllm/perspective/finetuning/scripts/get_embedding_finetuned.sh")
SAMPLE_SCRIPT_PATH = os.path.join(ROOT, "src/clusterllm/granularity/scripts/sample_pairs.sh")
SAMPLE_PY_PATH = os.path.join(ROOT, "src/clusterllm/granularity/sample_pairs.py")

print("========================================================")
print("🤖 DÉMARRAGE DU PROTOCOLE DE RÉPARATION AUTOMATIQUE")
print("========================================================")

def find_config():
    """Cherche config.json partout dans le dossier checkpoints"""
    print(f"🔍 Scan du dossier : {CHECKPOINT_ROOT}")
    for root, dirs, files in os.walk(CHECKPOINT_ROOT):
        if "config.json" in files:
            return os.path.join(root, "config.json"), root
    return None, None

def run_cmd(cmd):
    """Exécute une commande shell et arrête tout si elle échoue"""
    print(f"🚀 Exécution : {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"❌ ÉCHEC de la commande : {cmd}")
        sys.exit(1)

# ÉTAPE 1 : RECHERCHE DU MODÈLE
config_file, model_dir = find_config()

# ÉTAPE 1-BIS : SI MODÈLE ABSENT -> ON LE CRÉE
if not config_file:
    print("⚠️ Aucun modèle trouvé. Lancement automatique du Finetuning...")
    # On s'assure que le script de finetuning est exécutable
    run_cmd(f"chmod +x {FINETUNE_SCRIPT}")
    run_cmd(FINETUNE_SCRIPT)
    
    # On re-cherche après l'entraînement
    config_file, model_dir = find_config()
    if not config_file:
        print("❌ ERREUR CRITIQUE : Le finetuning a échoué à produire un config.json.")
        sys.exit(1)

print(f"✅ Modèle localisé ici : {model_dir}")

# ÉTAPE 2 : RÉPARATION DE L'IDENTITÉ (MPNET)
with open(config_file, 'r') as f:
    data = json.load(f)

if data.get("model_type") != "mpnet":
    print("🔧 Réparation de l'identité du modèle (mpnet)...")
    data["model_type"] = "mpnet"
    with open(config_file, 'w') as f:
        json.dump(data, f, indent=2)
else:
    print("✅ Identité du modèle OK.")

# ÉTAPE 3 : PATCH DU CODE PYTHON (Text vs Input)
# On lit le fichier sample_pairs.py pour être sûr qu'il est correct
with open(SAMPLE_PY_PATH, 'r') as f:
    code = f.read()
if "d['input']" in code:
    print("🔧 Correction du bug 'input' -> 'text' dans sample_pairs.py...")
    code = code.replace("d['input']", "d['text']")
    with open(SAMPLE_PY_PATH, 'w') as f:
        f.write(code)

# ÉTAPE 4 : GÉNÉRATION DU SCRIPT D'EMBEDDING AVEC LE BON CHEMIN
print("📝 Configuration du script d'embedding...")
embed_content = f"""#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="{ROOT}"
export PYTHONPATH="${{REPO_ROOT}}/src:${{PYTHONPATH:-}}"
cd "${{REPO_ROOT}}/src/clusterllm/perspective/finetuning"

python get_embedding.py \\
    --task_name "banking77" \\
    --data_path "${{REPO_ROOT}}/datasets/banking77/test.jsonl" \\
    --result_file "${{REPO_ROOT}}/datasets/banking77/embeddings_finetuned.pkl" \\
    --model_name "{model_dir}" \\
    --batch_size 32 \\
    --overwrite
"""
with open(EMBED_SCRIPT_PATH, 'w') as f:
    f.write(embed_content)
run_cmd(f"chmod +x {EMBED_SCRIPT_PATH}")

# ÉTAPE 5 : EXÉCUTION DE LA CHAÎNE
print("---------- LANCEMENT DE LA GÉNÉRATION DES EMBEDDINGS ----------")
run_cmd(EMBED_SCRIPT_PATH)

print("---------- LANCEMENT DE L'ÉCHANTILLONNAGE (SAMPLE PAIRS) ----------")
# On s'assure que le script sample pointe vers le bon fichier pkl
# (Le script shell sample_pairs.sh que nous avons fait précédemment pointe déjà vers embeddings_finetuned.pkl)
run_cmd(f"chmod +x {SAMPLE_SCRIPT_PATH}")
run_cmd(SAMPLE_SCRIPT_PATH)

print("========================================================")
print("🎉 MISSION ACCOMPLIE : TOUT EST GÉNÉRÉ CORRECTEMENT")
print("========================================================")
