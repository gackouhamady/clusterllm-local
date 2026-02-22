#!/usr/bin/env python3
import os
import json
import h5py
import torch
import logging
import argparse
import numpy as np
from pathlib import Path

from InstructorEmbedding import INSTRUCTOR
from clustering_utils.evaluator import ClusteringEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _read_jsonl(data_path: str, measure: bool):
    """
    Robust JSONL loader:
    - supports both keys: 'input' (original) or 'text' (your pipeline)
    - if measure=True, requires 'label'
    """
    texts, labels = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON at {data_path}:{line_no}: {e}")

            if "input" in obj:
                txt = obj["input"]
            elif "text" in obj:
                txt = obj["text"]
            else:
                raise ValueError(
                    f"Missing 'input' or 'text' field at {data_path}:{line_no}. "
                    f"Available keys: {list(obj.keys())}"
                )

            texts.append(txt)

            if measure:
                if "label" not in obj:
                    raise ValueError(f"Label not provided at {data_path}:{line_no}")
                labels.append(obj["label"])

    if measure:
        return texts, labels
    return texts, None


def _resolve_checkpoint(checkpoint: str | None) -> tuple[Path | None, str | None]:
    """
    Returns:
      (path_to_weights_file_or_dir, mode)
      mode in {"safetensors", "bin"} or None
    Accepts:
      --checkpoint /path/to/dir
      --checkpoint /path/to/model.safetensors
      --checkpoint /path/to/pytorch_model.bin
    """
    if checkpoint is None:
        return None, None

    ckpt = Path(checkpoint).expanduser()

    # If user passed a file directly
    if ckpt.is_file():
        if ckpt.suffix == ".safetensors":
            return ckpt, "safetensors"
        if ckpt.name.endswith(".bin"):
            return ckpt, "bin"
        # unknown file type -> try load as torch
        return ckpt, "bin"

    # If user passed a directory
    if ckpt.is_dir():
        safe_path = ckpt / "model.safetensors"
        bin_path = ckpt / "pytorch_model.bin"
        if safe_path.exists():
            return safe_path, "safetensors"
        if bin_path.exists():
            return bin_path, "bin"

        # Some trainers save as "pytorch_model.bin.index.json" + shards, etc.
        # In that case, SentenceTransformer/AutoModel should load from dir,
        # but your pipeline expects a single file. We fail clearly.
        raise FileNotFoundError(
            f"No checkpoint weights found in directory: {ckpt}\n"
            f"Expected one of:\n"
            f"  - {safe_path}\n"
            f"  - {bin_path}\n"
        )

    raise FileNotFoundError(f"--checkpoint path not found: {ckpt}")


def _load_state_dict(weights_path: Path, mode: str) -> dict:
    if mode == "safetensors":
        try:
            from safetensors.torch import load_file as safe_load_file
        except Exception as e:
            raise RuntimeError(
                "You are trying to load a .safetensors checkpoint but 'safetensors' "
                f"is not available. Install it in your env. Details: {e}"
            )
        return safe_load_file(str(weights_path))

    # mode == "bin"
    return torch.load(str(weights_path), map_location="cpu")


def main():
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
    parser.add_argument("--measure", action="store_true", help="if measure clustering performance")
    parser.add_argument("--overwrite", action="store_true", help="if overwrite the embedding")
    args = parser.parse_args()

    if not args.data_path:
        raise ValueError("--data_path is required")
    if not args.result_file:
        raise ValueError("--result_file is required")
    if not args.model_name:
        raise ValueError("--model_name is required")

    _ensure_parent_dir(args.result_file)

    texts, labels = _read_jsonl(args.data_path, measure=args.measure)

    # If embeddings already exist and overwrite is false -> just eval_only
    if os.path.exists(args.result_file) and not args.overwrite:
        with h5py.File(args.result_file, "r") as f:
            embeds = np.asarray(f["embeds"])
        evaluator = ClusteringEvaluator(sentences=texts, labels=labels, args=args)
        measures = evaluator.eval_only(embeds)
        print(measures)
        print("--DONE--")
        return

    # Build INSTRUCTOR model (paper-style)
    model = INSTRUCTOR(args.model_name, cache_folder=args.cache_dir)

    # Optional: load finetuned weights
    weights_path, mode = _resolve_checkpoint(args.checkpoint)
    if weights_path is not None:
        logger.info("Loading checkpoint weights from %s (%s)", weights_path, mode)
        state_dict = _load_state_dict(weights_path, mode)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys when loading checkpoint (strict=False): %s", missing[:20])
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint (strict=False): %s", unexpected[:20])

    # Prompt handling (keeps original behavior)
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
        _ensure_parent_dir(measures_path)
        with open(measures_path, "w", encoding="utf-8") as f:
            json.dump(measures, f)

    print(measures)
    print("--DONE--")


if __name__ == "__main__":
    main()
