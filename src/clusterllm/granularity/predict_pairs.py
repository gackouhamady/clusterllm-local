import argparse
import json
import os
from tqdm import tqdm

from tools import (
    add_ollama_cli_args,
    build_ollama_client_from_args,
    delayed_completion,
)

def prepare_data(prompt, datum):
    postfix = "\n\nPlease respond with 'Yes' or 'No' without explanation."
    input_txt = datum["input"]
    return prompt + "\n" + input_txt + postfix

def post_process(completion):
    try:
        content = completion["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        content = ""
    result = []
    if "Yes" in content and "No" not in content:
        result.append("Yes")
    elif "No" in content and "Yes" not in content:
        result.append("No")
    return content, result

def predict(args):
    # This now uses the recognized 'llama3_q4km' nickname
    client = build_ollama_client_from_args(args)
    
    # Safely get the model nickname for the filename
    # argparse converts '--ollama-model' to 'ollama_model'
    model_nick = getattr(args, 'ollama_model', 'ollama')
    
    prompt_file_name = args.prompt_file.split("/")[-1].split(".")[0]
    pred_path = args.data_path.split("/")[-1].replace(".json", f"-{model_nick}-{prompt_file_name}.json")
    pred_path = os.path.join("predicted_pair_results", pred_path)
    os.makedirs("predicted_pair_results", exist_ok=True)

    print(f"Saving results to: {pred_path}")

    with open(args.prompt_file, "r") as f:
        prompts = json.load(f)
        task_prompt = prompts[args.dataset]

    with open(args.data_path, "r") as f:
        raw_data = json.load(f)
        data = raw_data["test_inputs"] if isinstance(raw_data, dict) else raw_data

    for idx, datum in tqdm(enumerate(data), total=len(data)):
        if "prediction" in datum and not args.overwrite:
            continue

        prompt = prepare_data(task_prompt, datum)
        
        if idx == 0:
            print("\n--- FIRST PROMPT PREVIEW ---")
            print(prompt)
            print("----------------------------\n")

        completion, error = delayed_completion(
            client=client,
            prompt=prompt,
            delay_in_seconds=float(args.delay),
            max_trials=int(args.max_trials),
        )

        if completion is None:
            print(f"Error at index {idx}: {error}")
            continue

        content, results = post_process(completion)
        data[idx]["content"] = content
        data[idx]["prediction"] = results

        if idx % args.save_every == 0:
            with open(pred_path, "w") as f:
                json.dump({"test_inputs": data}, f, indent=4)

    with open(pred_path, "w") as f:
        json.dump({"test_inputs": data}, f, indent=4)
    print("✅ Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--max_trials", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    
    # Use the helper from tools.py to add --ollama-model and others
    add_ollama_cli_args(parser)
    
    args = parser.parse_args()
    predict(args)