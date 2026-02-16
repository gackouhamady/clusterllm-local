import os
import argparse
from datasets import load_dataset

def load_and_save_dataset(dataset_name: str, repo_root: str = ".") -> None:
    # Nettoyage du nom pour le dossier
    clean_name = dataset_name.split("/")[-1]
    output_dir = os.path.join(repo_root, "datasets", clean_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📥 Downloading dataset: '{dataset_name}'...")
    try:
        dataset = load_dataset(dataset_name)
        splits = dataset.keys()
        print(f"Detected splits: {list(splits)}")

        for split_name in splits:
            # Sauvegarde normale
            target_path = os.path.join(output_dir, f"{split_name}.jsonl")
            dataset[split_name].to_json(target_path, orient="records", lines=True)
            
            # --- ASTUCE POUR LE SCRIPT BASH ---
            # Si c'est le seul split de test/validation, on assure qu'un 'test.jsonl' existe
            if split_name in ["test", "validation", "valid"] and not os.path.exists(os.path.join(output_dir, "test.jsonl")):
                symlink_path = os.path.join(output_dir, "test.jsonl")
                dataset[split_name].to_json(symlink_path, orient="records", lines=True)
                print(f"👉 Force created 'test.jsonl' from '{split_name}' for compatibility.")

        print(f"✅ Dataset {clean_name} ready in {output_dir}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    args = parser.parse_args()
    load_and_save_dataset(args.name)