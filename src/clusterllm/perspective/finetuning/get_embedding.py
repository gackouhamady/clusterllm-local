import os
import json
import h5py
import torch
import logging
import argparse
import numpy as np
from InstructorEmbedding import INSTRUCTOR
from clustering_utils.evaluator import ClusteringEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=None, type=str)
parser.add_argument("--task_name", default=None, type=str)
parser.add_argument("--data_path", default=None, type=str)
parser.add_argument("--cache_dir", default=None, type=str)
parser.add_argument("--result_file", default=None, type=str)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--batch_size", default=-1, type=int)
parser.add_argument("--checkpoint", default=None, type=str)
parser.add_argument("--scale", default="small", type=str)
parser.add_argument(
    "--measure",
    action="store_true",
    help="if measure clustering performance",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="if overwrite the embedding",
)
args = parser.parse_args()

if not args.data_path:
    raise ValueError("--data_path is required")
if not args.result_file:
    raise ValueError("--result_file is required")
if not args.model_name:
    raise ValueError("--model_name is required")

# Ensure output dir exists (DVC-friendly)
os.makedirs(os.path.dirname(os.path.abspath(args.result_file)), exist_ok=True)

with open(args.data_path, "r", encoding="utf-8") as f:
    data = [json.loads(l) for l in f]

texts, labels = [], []
for datum in data:
    texts.append(datum["text"])
    if args.measure:
        if "label" in datum:
            labels.append(datum["label"])
        else:
            raise ValueError("Label not provided!")
    else:
        labels = None

if os.path.exists(args.result_file) and not args.overwrite:
    with h5py.File(args.result_file, "r") as f:
        embeds = np.asarray(f["embeds"])
    evaluator = ClusteringEvaluator(sentences=texts, labels=labels, args=args)
    measures = evaluator.eval_only(embeds)

else:
    model = INSTRUCTOR(args.model_name, cache_folder=args.cache_dir)

    if args.checkpoint is not None:
        logger.info("Loading from %s ...", args.checkpoint)
        state_path = os.path.join(args.checkpoint, "pytorch_model.bin")
        state_dict = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state_dict)

    if args.prompt is None:
        args.prompt = args.model_name

    if args.prompt not in ["hkunlp/instructor-xl", "hkunlp/instructor-base"]:
        args.prompt = "hkunlp/instructor-large"

    evaluator = ClusteringEvaluator(sentences=texts, labels=labels, args=args)
    measures, embeds = evaluator(model)

    with h5py.File(args.result_file, "w") as f:
        f.create_dataset("embeds", data=embeds)

if measures is not None and args.measure:
    measures_path = args.result_file.replace(".hdf5", "_measures.json")
    with open(measures_path, "w", encoding="utf-8") as f:
        json.dump(measures, f)

print(measures)
print("--DONE--")
