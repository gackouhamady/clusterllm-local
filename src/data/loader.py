import os
import pandas as pd
from datasets import load_dataset

def load_bank77():
    print("📥 Téléchargement de Bank77...")
    # Dataset officiel
    dataset = load_dataset("banking77")
    
    # Création du dossier
    os.makedirs("data/raw", exist_ok=True)
    
    # Sauvegarde CSV
    train_path = "data/raw/bank77_train.csv"
    test_path = "data/raw/bank77_test.csv"
    
    pd.DataFrame(dataset['train']).to_csv(train_path, index=False)
    pd.DataFrame(dataset['test']).to_csv(test_path, index=False)
    
    print(f"✅ Fichiers générés : {train_path}, {test_path}")

if __name__ == "__main__":
    load_bank77()