# the model takes in a pair of datapoints
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
    return prompt + input_txt + postfix


def post_process(completion):
    content = completion["choices"][0]["message"]["content"].strip()
    result = []
    if "Yes" in content and "No" not in content:
        result.append("Yes")
    elif "No" in content and "Yes" not in content:
        result.append("No")
    return content, result


def predict(args):
    # ---- STRICT LOCAL: build Ollama client from CLI/env ----
    client = build_ollama_client_from_args(args)

    prompt_file_name = args.prompt_file.split("/")[-1].split(".")[0]

    # name output using ollama model tag (instead of args.model_name)
    model_tag = args.ollama_model or os.getenv("CLUSTERLLM_OLLAMA_MODEL_KEY", "ollama")
    pred_path = args.data_path.split("/")[-1].replace(".json", f"-{model_tag}-{prompt_file_name}.json")
    pred_path = os.path.join("predicted_pair_results", pred_path)

    print("Save in:", pred_path)

    num_clusters = None

    # ---- load data (resume if exists) ----
    if os.path.exists(pred_path):
        with open(pred_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                if "num_clusters" in data:
                    num_clusters = data["num_clusters"]
                data = data["test_inputs"]
    else:
        with open(args.data_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                if "num_clusters" in data:
                    num_clusters = data["num_clusters"]
                data = data["test_inputs"]

    # ---- load previous predictions if provided ----
    if args.previous_path is not None:
        with open(args.previous_path, "r") as f:
            prev_data = json.load(f)
            if isinstance(prev_data, dict):
                prev_data = prev_data["test_inputs"]
        prev_inputs = {d["input"]: d for d in prev_data}
    else:
        prev_inputs = {}

    # ---- load prompts ----
    with open(args.prompt_file, "r") as f:
        prompts = json.load(f)
        task_prompt = prompts[args.dataset]

    # ---- prepare prompts ----
    for d in data:
        if "prepared" not in d:
            d["prepared"] = prepare_data(task_prompt, d)

    # ---- inference loop ----
    for idx, datum in tqdm(enumerate(data), total=len(data)):
        if idx == 0:
            print(datum["prepared"])

        if "prediction" in datum:
            continue

        # reuse previous predictions if same input exists
        if datum["input"] in prev_inputs:
            data[idx]["content"] = prev_inputs[datum["input"]]["content"]
            data[idx]["prediction"] = prev_inputs[datum["input"]]["prediction"]
            continue

        prompt = datum["prepared"]

        completion, error = delayed_completion(
            client=client,
            prompt=prompt,
            delay_in_seconds=float(args.delay),
            max_trials=int(args.max_trials),
        )

        if completion is None:
            print(f"Saving data after {idx + 1} inference.")
            with open(pred_path, "w") as f:
                out = data
                if num_clusters is not None:
                    out = {"num_clusters": num_clusters, "test_inputs": data}
                json.dump(out, f)
            print(error)
            raise RuntimeError(f"Ollama completion failed at idx={idx}") from error

        content, results = post_process(completion)
        data[idx]["content"] = content
        data[idx]["prediction"] = results

        if idx % args.save_every == 0 and idx > 0:
            print(f"Saving data after {idx + 1} inference.")
            with open(pred_path, "w") as f:
                out = data
                if num_clusters is not None:
                    out = {"num_clusters": num_clusters, "test_inputs": data}
                json.dump(out, f)

    # ---- final save ----
    with open(pred_path, "w") as f:
        out = data
        if num_clusters is not None:
            out = {"num_clusters": num_clusters, "test_inputs": data}
        json.dump(out, f)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--data_path", default=None, type=str, required=True)

    # STRICT LOCAL: remove OpenAI args
    # parser.add_argument("--openai_org", ...)  # removed
    # parser.add_argument("--model_name", ...)  # removed

    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max_trials", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--num_responses", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)  # kept for compatibility, not used by local client
    parser.add_argument("--previous_path", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, required=True)

    # Add LOCAL Ollama flags
    add_ollama_cli_args(parser)

    args = parser.parse_args()
    predict(args)
