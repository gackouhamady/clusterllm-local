import h5py
import numpy as np
import os

base_path = "/home/hamadygackou777/clusterllm-local/datasets/banking77"
orig_path = os.path.join(base_path, "embeddings.pkl")
tuned_path = os.path.join(base_path, "embeddings_finetuned.h5")

def get_data_and_key(path):
    if not os.path.exists(path):
        return None, f"❌ Fichier introuvable: {path}"
    try:
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            if not keys:
                return None, f"❌ Aucune clé trouvée dans {os.path.basename(path)}"
            # On utilise la première clé trouvée (souvent 'feat' ou 'features')
            first_key = keys[0]
            data = f[first_key][0]
            return data, first_key
    except Exception as e:
        return None, f"❌ Erreur lors de la lecture de {os.path.basename(path)}: {e}"

print("\n--- EXPLORATION ET VALIDATION DES VECTEURS ---")

v_orig, key_orig = get_data_and_key(orig_path)
v_tuned, key_tuned = get_data_and_key(tuned_path)

if v_orig is None or v_tuned is None:
    print(v_orig if v_orig is None else "")
    print(v_tuned if v_tuned is None else "")
    exit()

print(f"✅ Fichier Initial : Clé détectée = '{key_orig}'")
print(f"✅ Fichier Finetuné : Clé détectée = '{key_tuned}'")

# Comparaison mathématique
distance = np.linalg.norm(v_orig - v_tuned)

print(f"\nInitial (5 prem. valeurs): {v_orig[:5]}")
print(f"Finetuné (5 prem. valeurs): {v_tuned[:5]}")

if distance > 1e-6:
    print(f"\n🚀 RÉSULTAT : Distance = {distance:.6f}")
    print("Succès ! Votre modèle Instructor a bien été modifié par le savoir de Llama 3.")
else:
    print(f"\n⚠️ RÉSULTAT : Distance nulle ({distance:.6f})")
    print("Les fichiers sont identiques. Le finetuning n'a pas été appliqué lors de l'extraction.")
