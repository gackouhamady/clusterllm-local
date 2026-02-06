import os
import pandas as pd
from datasets import load_dataset


def load_bank77():
    print("📥 Téléchargement de Bank77...")
    dataset = load_dataset("banking77")

    # On remonte de 2 niveaux pour atteindre la racine du projet depuis src/data/
    # Ou on utilise un chemin absolu/relatif simple par rapport à l'exécution
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    train_path = f"{output_dir}/bank77_train.csv"
    test_path = f"{output_dir}/bank77_test.csv"

    pd.DataFrame(dataset['train']).to_csv(train_path, index=False)
    pd.DataFrame(dataset['test']).to_csv(test_path, index=False)

    print(f"✅ Fichiers générés dans {output_dir}")


if __name__ == "__main__":
    load_bank77()
