import os
import json
import h5py
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- CONFIGURATION ---
# Chemins
REPO_ROOT = "/home/hamadygackou777/clusterllm-local"
DATASET = "banking77"
# Votre checkpoint réparé
MODEL_PATH = "./checkpoints/banking77_finetuned_instructor_qwen2.5_7b/checkpoint-738"
DATA_PATH = os.path.join(REPO_ROOT, "datasets", DATASET, "test.jsonl")
OUTPUT_PATH = os.path.join(REPO_ROOT, "datasets", DATASET, "embeddings_finetuned.h5")

print("========================================================")
print(f"🛠️  GÉNÉRATION MANUELLE D'EMBEDDINGS")
print(f"📍 Modèle  : {MODEL_PATH}")
print(f"📂 Données : {DATA_PATH}")
print(f"💾 Sortie  : {OUTPUT_PATH}")
print("========================================================")

# 1. Chargement des données
print("📖 Lecture du fichier JSONL...")
sentences = []
labels = []
try:
    with open(DATA_PATH, 'r') as f:
        for line in f:
            data = json.loads(line)
            sentences.append(data['text'])
            labels.append(data['label'])
    print(f"✅ {len(sentences)} phrases chargées.")
except FileNotFoundError:
    print(f"❌ ERREUR: Fichier introuvable {DATA_PATH}")
    exit(1)

# 2. Chargement du modèle
print("🧠 Chargement du modèle SentenceTransformer...")
try:
    model = SentenceTransformer(MODEL_PATH)
    print("✅ Modèle chargé avec succès sur GPU.")
except Exception as e:
    print(f"❌ ERREUR chargement modèle: {e}")
    exit(1)

# 3. Encodage (Batch size 256 pour aller vite)
print("🚀 Calcul des embeddings...")
embeddings = model.encode(
    sentences, 
    batch_size=256, 
    show_progress_bar=True, 
    convert_to_numpy=True,
    normalize_embeddings=True # Souvent nécessaire pour le clustering
)

print(f"📊 Forme des embeddings : {embeddings.shape}")

# 4. Sauvegarde HDF5
print(f"💾 Écriture dans {OUTPUT_PATH}...")
try:
    with h5py.File(OUTPUT_PATH, "w") as f:
        f.create_dataset("features", data=embeddings)
        # On sauvegarde aussi les labels pour vérification future si besoin
        f.create_dataset("labels", data=np.array(labels))
    print("✅ Sauvegarde terminée !")
except Exception as e:
    print(f"❌ ERREUR écriture fichier: {e}")

# 5. Vérification immédiate
print("🔍 Vérification du fichier...")
try:
    with h5py.File(OUTPUT_PATH, 'r') as f:
        keys = list(f.keys())
        shape = f['features'].shape
        print(f"🎉 SUCCÈS TOTAL. Le fichier est valide.")
        print(f"   Clés trouvées : {keys}")
        print(f"   Dimensions    : {shape}")
except Exception as e:
    print(f"❌ Le fichier semble toujours corrompu : {e}")


