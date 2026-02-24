"""
This follows exactly the pretraining scheme (CL contrastive objective for INSTRUCTOR),
but is made robust to any dataset/LLM-generated converted triplets.

Key robustness:
- Normalize the INSTRUCTOR instruction (cur_e[k][0]) instead of asserting a specific prefix.
- Keep the rest of the training logic identical.
"""

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
import datasets
import nltk
import transformers
from filelock import FileLock
from InstructorEmbedding import INSTRUCTOR
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# ---- transformers version compatibility (robust) ----
check_min_version("4.20.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install datasets>=1.8.0")

try:
    from transformers.utils import is_offline_mode  # transformers <5 and some 4.x
except Exception:
    def is_offline_mode():
        return False

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock"):
        nltk.download("punkt", quiet=True)

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


def has_length(dataset):
    try:
        return len(dataset) is not None
    except TypeError:
        return False


class InstructorTrainer(Seq2SeqTrainer):
    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is not None:
            self.train_dataset = train_dataset

        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        if self.args.world_size <= 1:
            return SequentialSampler(self.train_dataset)
        return DistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=seed,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Paper constraint: batch must be same task
        for task_id in inputs["task_name"]:
            assert task_id == inputs["task_name"][0], (
                "Examples in the same batch should come from the same task, "
                f"but task {task_id} and task {inputs['task_name'][0]} are found"
            )

        cur_results = {}
        for k in ["query", "pos", "neg"]:
            cur_inputs = {
                "input_ids": inputs[f"{k}_input_ids"],
                "attention_mask": inputs[f"{k}_attention_mask"],
                "context_masks": inputs[f"{k}_context_masks"],
            }
            cur_results[k] = model(cur_inputs)["sentence_embedding"]

        embeddings_query = cur_results["query"]
        embeddings_pos = cur_results["pos"]
        embeddings_neg = cur_results["neg"]

        num = len(embeddings_query)
        from torch import nn

        similarity_fct = nn.CosineSimilarity(dim=-1)
        temperature = self.args.cl_temperature if getattr(self.args, "cl_temperature", None) else 0.05

        # anchor=query, positives=pos, negatives=all neg in batch
        all_scores = None
        for i in range(num):
            anchor_emb = embeddings_query[i].unsqueeze(0)
            pos_emb = embeddings_pos[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / temperature
            for j in range(num):
                one_neg_emb = embeddings_neg[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            all_scores = cur_score.unsqueeze(0) if all_scores is None else torch.cat(
                [all_scores, cur_score.unsqueeze(0)], dim=0
            )

        labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
        loss = nn.CrossEntropyLoss()(all_scores, labels)

        # symmetry term (paper style)
        all_another_scores = None
        for i in range(num):
            anchor_emb = embeddings_pos[i].unsqueeze(0)
            pos_emb = embeddings_query[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / temperature
            for j in range(num):
                if i == j:
                    continue
                one_neg_emb = embeddings_query[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            all_another_scores = cur_score.unsqueeze(0) if all_another_scores is None else torch.cat(
                [all_another_scores, cur_score.unsqueeze(0)], dim=0
            )

        labels_another = torch.zeros(all_another_scores.size(0)).long().to(embeddings_query.device)
        loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)

        # ==========================
        # CHANGE (LOSS ONLY): add anti-collapse regularizers without touching the rest of the pipeline
        # (1) Triplet-margin regularizer (FaceNet-style)
        # (2) Variance anti-collapse regularizer (VICReg-style)
        # ==========================

        # Hyperparameters (kept local to avoid changing args/configs elsewhere)
        margin = 0.2          # gamma
        w_trip = 0.5          # weight for triplet-margin term
        w_var = 1.0           # weight for variance anti-collapse term
        var_target = 1.0      # target per-dimension std (VICReg-style)
        eps = 1e-4

        # (1) Triplet-margin: max(0, gamma + s(q,n) - s(q,p))
        if w_trip > 0:
            sim_qp = similarity_fct(embeddings_query, embeddings_pos)  # (B,)
            sim_qn = similarity_fct(embeddings_query, embeddings_neg)  # (B,)
            trip_loss = torch.relu(margin + sim_qn - sim_qp).mean()
            loss = loss + w_trip * trip_loss

        # (2) Variance anti-collapse: encourage non-zero variance per dimension across the batch
        if w_var > 0:
            z = torch.cat([embeddings_query, embeddings_pos, embeddings_neg], dim=0)  # (3B, d)
            std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)                     # (d,)
            var_loss = torch.relu(var_target - std).mean()
            loss = loss + w_var * var_loss

        return loss


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or HF identifier"})
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    resize_position_embeddings: Optional[bool] = field(default=None)
    init_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "A model checkpoint dir containing pytorch_model.bin or model.safetensors"},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    processed_data_dir: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None, metadata={"help": "JSON file with converted triplets"})
    overwrite_cache: bool = field(default=False)
    add_prompt_to_document: bool = field(default=True)
    preprocessing_num_workers: Optional[int] = field(default=None)
    max_examples: Optional[int] = field(default=None)
    cl_temperature: Optional[float] = field(default=0.05)
    max_source_length: Optional[int] = field(default=512)
    max_train_samples: Optional[int] = field(default=None)

    # Robust default instruction if provided instruction is empty/malformed
    default_instruction: str = field(
        default="Represent the text for clustering: ",
        metadata={"help": "Fallback INSTRUCTOR instruction when input instruction is invalid."},
    )

    def __post_init__(self):
        pass


def _normalize_instruction(instr: str, fallback: str) -> str:
    """
    Paper expects INSTRUCTOR style instruction often starting with 'Represent ...'.
    In practice, converted triplets may contain other prefixes.
    We keep the instruction field but enforce a safe, INSTRUCTOR-compatible fallback.
    """
    if instr is None:
        return ""
    instr = str(instr).lstrip()  # remove leading spaces/tabs
    if instr == "":
        return ""
    # Keep if already in the expected style
    if instr.lower().startswith("represent "):
        return instr
    # Otherwise, normalize to fallback
    return fallback


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Map temperature into training args (paper style)
    training_args.cl_temperature = data_args.cl_temperature
    training_args.remove_unused_columns = False

    os.makedirs(training_args.output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.ERROR
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    set_seed(training_args.seed)

    if data_args.train_file is None:
        raise ValueError("You must provide --train_file pointing to converted *-train.json")

    with open(data_args.train_file, "r", encoding="utf-8") as f:
        train_examples_raw = json.load(f)

    if data_args.max_examples is not None:
        train_examples_raw = train_examples_raw[: int(data_args.max_examples)]

    print(f"There are {len(train_examples_raw)} pairs to train in total")

    # Convert to DatasetDict like paper
    train_examples = {"query": [], "pos": [], "neg": [], "task_name": []}
    task_name_map = {}
    task_count = 0

    for cur_e in train_examples_raw:
        for k in ["query", "pos", "neg"]:
            for s in cur_e[k][:-1]:
                assert "!@#$%^&**!@#$%^&**" not in s
            cur_e[k][-1] = str(cur_e[k][-1])

            if not data_args.add_prompt_to_document:
                cur_e[k][0] = ""

            # Robust instruction normalization (instead of assert)
            cur_e[k][0] = _normalize_instruction(cur_e[k][0], data_args.default_instruction)

            train_examples[k].append("!@#$%^&**!@#$%^&**".join(cur_e[k]))

        if cur_e["task_name"] not in task_name_map:
            task_name_map[cur_e["task_name"]] = task_count
            task_count += 1
        train_examples["task_name"].append(task_name_map[cur_e["task_name"]])

    raw_datasets = DatasetDict({"train": Dataset.from_dict(train_examples)})

    model = INSTRUCTOR(model_args.model_name_or_path, cache_folder=model_args.cache_dir)
    column_names = raw_datasets["train"].column_names

    def preprocess_function(examples):
        all_tokenized = None
        for key in ["query", "pos", "neg"]:
            num = len(examples[key])
            contexts, concatenated = [], []
            for i in range(num):
                splits = examples[key][i].split("!@#$%^&**!@#$%^&**")
                assert len(splits) == 2
                contexts.append(splits[0])
                concatenated.append("".join(splits))

            tokenized = tokenizer(
                concatenated,
                padding="max_length",
                truncation="longest_first",
                return_tensors="pt",
                max_length=data_args.max_source_length,
            )
            context_tok = tokenizer(
                contexts,
                padding="max_length",
                truncation="longest_first",
                return_tensors="pt",
                max_length=data_args.max_source_length,
            )

            tokenized["context_masks"] = torch.sum(context_tok["attention_mask"], dim=1) - 1
            for my_idx in range(len(tokenized["context_masks"])):
                if tokenized["context_masks"][my_idx] <= 1:
                    tokenized["context_masks"][my_idx] = 0

            keys = list(tokenized.keys())
            if all_tokenized is None:
                all_tokenized = {}
            for kk in keys:
                all_tokenized[f"{key}_{kk}"] = tokenized[kk].tolist()

        all_tokenized["task_name"] = examples["task_name"]
        return all_tokenized

    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    label_pad_token_id = -100 if data_args.overwrite_cache else tokenizer.pad_token_id
    data_collator = default_data_collator

    trainer = InstructorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=None,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Optional init checkpoint (bin or safetensors)
    if model_args.init_checkpoint is not None:
        ckpt_bin = os.path.join(model_args.init_checkpoint, "pytorch_model.bin")
        ckpt_safe = os.path.join(model_args.init_checkpoint, "model.safetensors")
        if os.path.exists(ckpt_bin):
            print(f"Loading from {ckpt_bin} ...")
            state_dict = torch.load(ckpt_bin, map_location="cpu")
            model.load_state_dict(state_dict)
        elif os.path.exists(ckpt_safe):
            print(f"Loading from {ckpt_safe} ...")
            from safetensors.torch import load_file as safe_load_file
            state_dict = safe_load_file(ckpt_safe)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(
                f"init_checkpoint set but no pytorch_model.bin/model.safetensors in {model_args.init_checkpoint}"
            )

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
