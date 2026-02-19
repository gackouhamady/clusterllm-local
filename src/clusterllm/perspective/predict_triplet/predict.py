import argparse
import json
import os
import glob
from tqdm import tqdm
import concurrent.futures

from tools import (
    add_ollama_cli_args,
    build_ollama_client_from_args,
    delayed_completion,
    prepare_data,
    post_process,
)

def predict(args):
    # ---- 1. Handle Model Name Compatibility ----
    # Ensure args.ollama_model is populated even if passed as --model_name
    if not args.ollama_model and args.model_name:
        args.ollama_model = args.model_name

    # ---- 2. Build Client ----
    client = build_ollama_client_from_args(args)

    # ---- 3. Resolve Input File ----
    # If data_path is not explicitly provided, find the first json in input_dir matching the dataset
    input_file = args.data_path
    if not input_file and args.input_dir:
        # Search for json files that contain the dataset name (e.g. "bank77")
        search_pattern = os.path.join(args.input_dir, f"*{args.dataset}*.json")
        candidates = glob.glob(search_pattern)
        if not candidates:
            # Fallback: take any json
            candidates = glob.glob(os.path.join(args.input_dir, "*.json"))
        
        if candidates:
            input_file = candidates[0]
            print(f"Found input file: {input_file}")
        else:
            raise FileNotFoundError(f"No input JSON file found in {args.input_dir} for dataset {args.dataset}")
    elif not input_file:
        raise ValueError("Either --data_path or --input_dir must be provided.")

    # ---- 4. Construct Output Path ----
    # Naming convention: {original_name}-{model}-pred.json
    model_tag = args.ollama_model or "ollama"
    # Remove characters that might mess up filenames (like :)
    safe_model_tag = model_tag.replace(":", "_")
    
    temp_tag = f"-temp{round(args.ollama_temperature, 1)}" if args.ollama_temperature and args.ollama_temperature > 0 else ""
    
    base_name = os.path.basename(input_file)
    output_filename = base_name.replace(".json", f"-{safe_model_tag}{temp_tag}-pred.json")
    
    # Use output_dir if provided, otherwise default to "predicted_triplet_results"
    output_dir = args.output_dir or "predicted_triplet_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    pred_path = os.path.join(output_dir, output_filename)
    print("Save in:", pred_path)

    # ---- 5. Resume or Load Data ----
    if os.path.exists(pred_path):
        print("Resuming from existing prediction file...")
        with open(pred_path, "r") as f:
            data = json.load(f)
    else:
        with open(input_file, "r") as f:
            data = json.load(f)

    # ---- 6. Load Prompts ----
    # Ensure prompts.json is found relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_path = os.path.join(script_dir, "prompts.json")
    
    if not os.path.exists(prompts_path):
        # Fallback for current working directory
        prompts_path = "prompts.json"

    with open(prompts_path, "r") as f:
        prompts = json.load(f)
        if args.dataset not in prompts:
             raise ValueError(f"Dataset '{args.dataset}' not found in prompts.json")
        task_prompt = prompts[args.dataset]

    # Prepare prompt once
    for d in data:
        if "prepared" not in d:
            d["prepared"] = prepare_data(task_prompt, d)

# ---- 7. Inference Loop (MULTI-THREADING) ----
    if len(data) > 0 and "prepared" in data[0]:
        print(f"Sample Prompt:\n{data[0]['prepared']}\n")

    def process_item(idx, datum):
        # Si déjà prédit lors d'une exécution précédente, on passe
        if "prediction" in datum:
            return idx, True

        prompt = datum["prepared"]

        completion, error = delayed_completion(
            client=client,
            prompt=prompt,
            delay_in_seconds=float(args.delay),
            max_trials=int(args.max_trials),
            model=args.ollama_model 
        )

        if completion is None:
            print(f"Error at index {idx}: {error}")
            return idx, False

        content, results = post_process(completion, datum["options"])
        datum["content"] = content
        datum["prediction"] = results
        return idx, True

    max_workers = 4 # Parallélisme (doit correspondre à OLLAMA_NUM_PARALLEL)
    save_counter = 0

    print(f"🚀 Lancement de l'inférence avec {max_workers} requêtes en parallèle...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prépare toutes les tâches
        futures = {executor.submit(process_item, i, d): i for i, d in enumerate(data)}
        
        # Traite les tâches au fur et à mesure qu'elles se terminent
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
            idx = futures[future]
            success = future.result()

            if not success:
                print(f"Sauvegarde des données après échec à l'index {idx}.")
                with open(pred_path, "w") as f:
                    json.dump(data, f, indent=2)
                raise RuntimeError(f"Ollama completion failed at idx={idx}")

            save_counter += 1
            if save_counter >= args.save_every:
                with open(pred_path, "w") as f:
                    json.dump(data, f, indent=2)
                save_counter = 0

    # Sauvegarde finale
    with open(pred_path, "w") as f:
        json.dump(data, f, indent=2)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Core arguments
    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--data_path", default=None, type=str, help="Specific path to input json (optional if input_dir is set)")
    
    # New arguments for DVC path handling
    parser.add_argument("--input_dir", default=None, type=str, help="Directory containing input json files")
    parser.add_argument("--output_dir", default=None, type=str, help="Directory to save output json files")
    
    # Model name alias (for compatibility with bash script)
    parser.add_argument("--model_name", default=None, type=str, help="Alias for ollama_model")

    # Inference settings
    parser.add_argument("--delay", type=float, default=0.0) # Faster local inference
    parser.add_argument("--max_trials", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--num_responses", type=int, default=1)

    # Add Ollama config flags (host, model, temp, etc.)
    add_ollama_cli_args(parser)

    args = parser.parse_args()
    predict(args)