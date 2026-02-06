import argparse
import json
import os

from tqdm import tqdm

from tools import (
    add_ollama_cli_args,
    build_ollama_client_from_args,
    delayed_completion,
    prepare_data,
    post_process,
)


def predict(args):
    # ---- STRICT LOCAL: build Ollama client from CLI/env ----
    client = build_ollama_client_from_args(args)

    # ---- output path naming: keep your old convention, but use ollama model tag ----
    model_tag = args.ollama_model or os.getenv("CLUSTERLLM_OLLAMA_MODEL_KEY", "ollama")
    temp_tag = f"-temp{round(args.ollama_temperature, 1)}" if args.ollama_temperature and args.ollama_temperature > 0 else ""
    pred_path = args.data_path.split("/")[-1].replace(".json", f"-{model_tag}{temp_tag}-pred.json")
    pred_path = os.path.join("predicted_triplet_results", pred_path)

    print("Save in:", pred_path)

    # resume if exists
    if os.path.exists(pred_path):
        with open(pred_path, "r") as f:
            data = json.load(f)
    else:
        with open(args.data_path, "r") as f:
            data = json.load(f)

    # load prompt template
    with open("prompts.json", "r") as f:
        prompts = json.load(f)
        task_prompt = prompts[args.dataset]

    # prepare prompt once
    for d in data:
        if "prepared" not in d:
            d["prepared"] = prepare_data(task_prompt, d)

    # inference loop
    for idx, datum in tqdm(enumerate(data), total=len(data)):
        if idx == 0:
            print(datum["prepared"])

        if "prediction" in datum:
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
                json.dump(data, f)
            print(error)
            raise RuntimeError(f"Ollama completion failed at idx={idx}") from error

        content, results = post_process(completion, datum["options"])
        data[idx]["content"] = content
        data[idx]["prediction"] = results

        if idx % args.save_every == 0 and idx > 0:
            print(f"Saving data after {idx + 1} inference.")
            with open(pred_path, "w") as f:
                json.dump(data, f)

    with open(pred_path, "w") as f:
        json.dump(data, f)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # original args (keep dataset & data_path)
    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--data_path", default=None, type=str, required=True)

    # remove OpenAI args (STRICT LOCAL). We keep no --openai_org and no --model_name.
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max_trials", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--num_responses", type=int, default=1)

    # add Ollama config flags
    add_ollama_cli_args(parser)

    args = parser.parse_args()
    predict(args)
