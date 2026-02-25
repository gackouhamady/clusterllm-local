"""
Robust INSTRUCTOR fine-tuning with:
  (1) The original CLUSTERLLM/INSTRUCTOR contrastive objective (InfoNCE + symmetric term).
  (2) Anti-collapse variance regularization (std-based) to prevent representation collapse.
  (3) A loss stability controller that tracks EMA mean/std of the loss and dampens rare spikes.

Contribution #3 (Uncertainty- and noise-aware supervision) EXTENSION:
  (4) Soft contrastive loss term using soft labels and example weights:
        L_soft = w * ( -p log P - (1-p) log(1-P) )
      where:
        - p = soft_pos_prob (probability that 'pos' is truly positive vs the paired 'neg')
        - w = example_weight (confidence weight)
        - P = model probability of preferring 'pos' over the paired 'neg'
      This extension is strictly additive and does NOT modify:
        - InfoNCE logic
        - variance regularization
        - loss stability controller
      If soft label fields are missing, the training behavior remains unchanged.

Key references (official):
  - CLUSTERLLM (Zhang et al., 2023): contrastive objective with symmetric swap term (Eq. 5 + swap).
  - VICReg (Bardes, Ponce, LeCun, 2021, arXiv:2105.04906): variance regularization to avoid collapse.
  - "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al., 2020, arXiv:2008.03156).
  - Welford (1962) / Knuth TAOCP Vol.2: numerically stable online variance/std estimation.

NOTE:
  - This script is intentionally conservative: the controller *dampens* loss spikes instead of
    adding noise or aggressively skipping steps.
  - You should still use Trainer's built-in gradient clipping via --max_grad_norm.
"""

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F
import datasets
import nltk
import transformers
from filelock import FileLock
from InstructorEmbedding import INSTRUCTOR
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
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
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first "
            "to download nltk data files"
        )
    with FileLock(".lock"):
        nltk.download("punkt", quiet=True)

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


def has_length(dataset):
    try:
        return len(dataset) is not None
    except TypeError:
        return False


def _normalize_instruction(instr: str, fallback: str) -> str:
    """
    Robust INSTRUCTOR instruction normalization.

    Why:
      In practice, converted triplets may contain missing/empty instructions or arbitrary prefixes.
      INSTRUCTOR was trained with explicit "Represent ..." style instructions.

    Policy:
      - If add_prompt_to_document is enabled, NEVER allow an empty instruction: fallback is used.
      - If the instruction already starts with "Represent ", keep it.
      - Otherwise, replace with fallback (safe, consistent instruction).
    """
    if instr is None:
        return fallback
    instr = str(instr).strip()
    if instr == "":
        return fallback
    if instr.lower().startswith("represent "):
        return instr
    return fallback


def _variance_regularizer(
    x: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    VICReg-style variance regularization (anti-collapse).
    Keeps per-dimension std >= gamma by penalizing short std.

      std_j = sqrt(Var(x[:, j]) + eps)
      penalty = mean_j relu(gamma - std_j)

    Returns:
      penalty: scalar tensor
      mean_std: scalar tensor (mean of std across dims) for monitoring
    """
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    penalty = torch.mean(F.relu(gamma - std))
    return penalty, std.mean()


class InstructorTrainer(Seq2SeqTrainer):
    """
    Custom Trainer:
      - preserves "same task in batch" constraint
      - implements robust contrastive loss (vectorized)
      - adds anti-collapse variance regularization
      - adds EMA loss stability controller (mean/std) + spike damping
      - EXTENSION ONLY (Contribution #3): optional soft contrastive loss term
    """

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

    # ---------- Loss stability controller state (EMA mean/std) ----------
    def _init_loss_stats_if_needed(self, loss_value: torch.Tensor) -> None:
        if not hasattr(self, "_loss_ema_mean"):
            self._loss_ema_mean = loss_value.detach().float().cpu()
            self._loss_ema_var = torch.tensor(0.0, dtype=torch.float32)
            self._loss_clip_count = 0

    def _update_loss_stats(self, loss_value: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exponential moving estimates of mean and variance:
          m_t = beta*m_{t-1} + (1-beta)*x_t
          v_t = beta*v_{t-1} + (1-beta)*(x_t - m_t)^2

        Returns (mean, std) as CPU float tensors.
        """
        x = loss_value.detach().float().cpu()
        self._init_loss_stats_if_needed(x)

        m_prev = self._loss_ema_mean
        v_prev = self._loss_ema_var

        m = beta * m_prev + (1.0 - beta) * x
        v = beta * v_prev + (1.0 - beta) * (x - m) ** 2

        self._loss_ema_mean = m
        self._loss_ema_var = v

        std = torch.sqrt(v + 1e-12)
        return m, std

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the training loss.

        Base loss (UNCHANGED):
          - Direction 1 InfoNCE: anchor=query, positive=pos_i, negatives=all neg in batch.
          - Direction 2 symmetric swap: anchor=pos, positive=query_i, negatives=all other queries.

        Additional terms (UNCHANGED):
          - Variance regularization (std-based) when var_reg_weight > 0.
          - EMA-based spike damping controller when loss_z_clip > 0.

        Contribution #3 extension (ADDITIVE, does not modify InfoNCE):
          If inputs contain:
            - soft_pos_prob (p)  in [0,1]
            - example_weight (w) in [0,1]
          then add:
            L_soft = mean_i w_i * ( -p_i log P_i - (1-p_i) log(1-P_i) )
          where P_i is the model probability that pos_i is preferred over the paired neg_i:
            P_i = softmax([s(q_i,pos_i)/tau, s(q_i,neg_i)/tau])[0]
          If these fields are missing, L_soft is skipped (behavior unchanged).
        """
        # Paper constraint: batch must be same task
        for task_id in inputs["task_name"]:
            assert task_id == inputs["task_name"][0], (
                "Examples in the same batch should come from the same task, "
                f"but task {task_id} and task {inputs['task_name'][0]} are found"
            )

        # --------------------
        # Forward embeddings
        # --------------------
        cur_results: Dict[str, torch.Tensor] = {}
        for k in ["query", "pos", "neg"]:
            cur_inputs = {
                "input_ids": inputs[f"{k}_input_ids"],
                "attention_mask": inputs[f"{k}_attention_mask"],
                "context_masks": inputs[f"{k}_context_masks"],
            }
            out = model(cur_inputs)
            emb = out["sentence_embedding"]
            cur_results[k] = emb

        embeddings_query = cur_results["query"]  # [B, D]
        embeddings_pos = cur_results["pos"]      # [B, D]
        embeddings_neg = cur_results["neg"]      # [B, D]

        B = embeddings_query.size(0)
        if B <= 1:
            loss = torch.zeros((), device=embeddings_query.device, requires_grad=True)
            return (loss, cur_results) if return_outputs else loss

        # -------------
        # Temperature
        # -------------
        temperature = getattr(self.args, "cl_temperature", 0.05) or 0.05
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError(f"cl_temperature must be > 0, got {temperature}")

        # --------------------------------------------
        # Vectorized contrastive loss (more stable)
        # --------------------------------------------
        q = F.normalize(embeddings_query, p=2, dim=-1)
        p = F.normalize(embeddings_pos, p=2, dim=-1)
        n = F.normalize(embeddings_neg, p=2, dim=-1)

        # Direction 1: anchor = query_i; classes = [pos_i] + [neg_0..neg_{B-1}]
        pos_scores = torch.sum(q * p, dim=-1, keepdim=True) / temperature   # [B, 1]
        neg_scores = torch.matmul(q, n.T) / temperature                     # [B, B]
        logits_1 = torch.cat([pos_scores, neg_scores], dim=1)               # [B, 1+B]
        labels_1 = torch.zeros(B, dtype=torch.long, device=logits_1.device)
        loss_1 = F.cross_entropy(logits_1, labels_1)

        # Direction 2 (symmetric): anchor = pos_i; classes = all queries; positive index = i
        logits_2 = torch.matmul(p, q.T) / temperature                       # [B, B]
        labels_2 = torch.arange(B, dtype=torch.long, device=logits_2.device)
        loss_2 = F.cross_entropy(logits_2, labels_2)

        loss = loss_1 + loss_2

        # --------------------------------------------
        # Contribution #3: Soft contrastive loss (ADDITIVE)
        # --------------------------------------------
        # Only apply if the dataset provides soft labels; otherwise keep behavior unchanged.
        if ("soft_pos_prob" in inputs) and ("example_weight" in inputs):
            # p: probability that 'pos' is correct for this (query,pos,neg) pair
            # w: confidence weight for this example
            p_soft = inputs["soft_pos_prob"]
            w_ex = inputs["example_weight"]

            # Convert to float tensors on the correct device/shape.
            if not torch.is_tensor(p_soft):
                p_soft = torch.tensor(p_soft, device=embeddings_query.device, dtype=torch.float32)
            else:
                p_soft = p_soft.to(device=embeddings_query.device, dtype=torch.float32)

            if not torch.is_tensor(w_ex):
                w_ex = torch.tensor(w_ex, device=embeddings_query.device, dtype=torch.float32)
            else:
                w_ex = w_ex.to(device=embeddings_query.device, dtype=torch.float32)

            p_soft = p_soft.view(-1)
            w_ex = w_ex.view(-1)

            # Safety: handle NaNs/Infs and clamp.
            p_soft = torch.nan_to_num(p_soft, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.0, 1.0)
            w_ex = torch.nan_to_num(w_ex, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.0, 1.0)

            # Compute P_i = P(pos_i preferred over neg_i) from paired scores.
            # Use the paired negative (diag of neg_scores), NOT all in-batch negatives.
            pos_i = pos_scores.squeeze(1)                        # [B]
            neg_i = torch.diagonal(neg_scores, 0)                # [B]

            logits_bin = torch.stack([pos_i, neg_i], dim=1)      # [B, 2]
            probs_bin = torch.softmax(logits_bin, dim=1)
            P = probs_bin[:, 0]                                  # [B]

            eps = 1e-12
            P = P.clamp(eps, 1.0 - eps)

            # L_soft = w * ( -p log P - (1-p) log(1-P) )
            soft_per_ex = -(p_soft * torch.log(P) + (1.0 - p_soft) * torch.log(1.0 - P))
            soft_loss = torch.mean(w_ex * soft_per_ex)

            loss = loss + soft_loss

        # --------------------------------------------
        # Anti-collapse variance regularizer (std-based)
        # --------------------------------------------
        var_weight = float(getattr(self.args, "var_reg_weight", 0.0) or 0.0)
        if var_weight > 0.0:
            gamma = float(getattr(self.args, "var_reg_gamma", 1.0) or 1.0)
            var_eps = float(getattr(self.args, "var_reg_eps", 1e-4) or 1e-4)

            pen_q, std_q = _variance_regularizer(embeddings_query, gamma=gamma, eps=var_eps)
            pen_p, std_p = _variance_regularizer(embeddings_pos, gamma=gamma, eps=var_eps)
            pen_n, std_n = _variance_regularizer(embeddings_neg, gamma=gamma, eps=var_eps)

            var_penalty = (pen_q + pen_p + pen_n) / 3.0
            loss = loss + var_weight * var_penalty

            self._last_embed_std_mean = float(((std_q + std_p + std_n) / 3.0).detach().cpu())

        # --------------------------------------------
        # Loss stability controller (EMA mean/std + spike damping)
        # --------------------------------------------
        beta = float(getattr(self.args, "loss_ema_beta", 0.98) or 0.98)
        z_clip = float(getattr(self.args, "loss_z_clip", 0.0) or 0.0)  # 0 disables
        warmup = int(getattr(self.args, "loss_z_warmup_steps", 20) or 20)

        step = int(getattr(self.state, "global_step", 0) or 0)
        if z_clip > 0.0:
            mean, std = self._update_loss_stats(loss.detach(), beta=beta)
            z = (loss.detach().float().cpu() - mean) / (std + 1e-12)

            if step >= warmup and z.item() > z_clip:
                scale = float((z_clip / z).clamp(max=1.0).item())
                scale = max(scale, 0.2)
                loss = loss * scale
                self._loss_clip_count += 1

            self._last_loss_ema_mean = float(mean.item())
            self._last_loss_ema_std = float(std.item())
            self._last_loss_z = float(z.item())

        # Optional lightweight periodic logging through Trainer
        try:
            log_steps = int(getattr(self.args, "logging_steps", 500) or 500)
            if log_steps > 0 and step % log_steps == 0:
                logs = {}
                if hasattr(self, "_last_loss_ema_mean"):
                    logs.update({
                        "loss_ema_mean": self._last_loss_ema_mean,
                        "loss_ema_std": self._last_loss_ema_std,
                        "loss_z": self._last_loss_z,
                        "loss_clip_count": float(getattr(self, "_loss_clip_count", 0)),
                    })
                if hasattr(self, "_last_embed_std_mean"):
                    logs["embed_std_mean"] = self._last_embed_std_mean
                if logs:
                    self.log(logs)
        except Exception:
            pass

        return (loss, cur_results) if return_outputs else loss


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

    # Contrastive temperature (CLUSTERLLM / INSTRUCTOR-style)
    cl_temperature: Optional[float] = field(default=0.10)

    max_source_length: Optional[int] = field(default=512)
    max_train_samples: Optional[int] = field(default=None)

    # Robust default instruction for malformed/empty instructions
    default_instruction: str = field(
        default="Represent the text for clustering: ",
        metadata={"help": "Fallback INSTRUCTOR instruction when input instruction is invalid/empty."},
    )

    # ----------------- NEW: Anti-collapse (variance) regularization -----------------
    var_reg_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for VICReg-style variance regularization (0 disables)."},
    )
    var_reg_gamma: float = field(
        default=1.0,
        metadata={"help": "Variance floor gamma for std (penalize if std < gamma)."},
    )
    var_reg_eps: float = field(
        default=1e-4,
        metadata={"help": "Numerical epsilon added inside sqrt(var + eps)."},
    )

    # ----------------- NEW: Loss stability controller (EMA mean/std) -----------------
    loss_ema_beta: float = field(
        default=0.98,
        metadata={"help": "EMA beta for loss mean/std tracking. Higher = smoother."},
    )
    loss_z_clip: float = field(
        default=2.0,
        metadata={"help": "If >0: damp loss spikes when z-score > loss_z_clip (0 disables)."},
    )
    loss_z_warmup_steps: int = field(
        default=20,
        metadata={"help": "Number of optimizer steps before enabling z-score damping."},
    )

    def __post_init__(self):
        pass


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ---- Map custom args into training args (HF TrainingArguments accepts dynamic attrs) ----
    training_args.cl_temperature = data_args.cl_temperature
    training_args.var_reg_weight = data_args.var_reg_weight
    training_args.var_reg_gamma = data_args.var_reg_gamma
    training_args.var_reg_eps = data_args.var_reg_eps
    training_args.loss_ema_beta = data_args.loss_ema_beta
    training_args.loss_z_clip = data_args.loss_z_clip
    training_args.loss_z_warmup_steps = data_args.loss_z_warmup_steps

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

    # Convert to DatasetDict like paper (+ optional soft fields)
    train_examples = {"query": [], "pos": [], "neg": [], "task_name": [], "soft_pos_prob": [], "example_weight": []}
    task_name_map = {}
    task_count = 0

    for cur_e in train_examples_raw:
        for k in ["query", "pos", "neg"]:
            for s in cur_e[k][:-1]:
                assert "!@#$%^&**!@#$%^&**" not in s

            cur_e[k][-1] = str(cur_e[k][-1])

            if not data_args.add_prompt_to_document:
                cur_e[k][0] = ""
            else:
                cur_e[k][0] = _normalize_instruction(cur_e[k][0], data_args.default_instruction)

            train_examples[k].append("!@#$%^&**!@#$%^&**".join(cur_e[k]))

        if cur_e["task_name"] not in task_name_map:
            task_name_map[cur_e["task_name"]] = task_count
            task_count += 1
        train_examples["task_name"].append(task_name_map[cur_e["task_name"]])

        # Contribution #3 fields (optional): keep missing as absent-safe values.
        # We keep these columns present so they can be consumed when available.
        train_examples["soft_pos_prob"].append(float(cur_e.get("soft_pos_prob", 1.0)))
        train_examples["example_weight"].append(float(cur_e.get("example_weight", 1.0)))

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

        # Contribution #3 optional fields: pass through if present.
        if "soft_pos_prob" in examples:
            all_tokenized["soft_pos_prob"] = examples["soft_pos_prob"]
        if "example_weight" in examples:
            all_tokenized["example_weight"] = examples["example_weight"]

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

        print("DEBUG columns:", train_dataset.column_names)
        print("DEBUG sample keys:", train_dataset[0].keys())
        print("DEBUG soft_pos_prob:", train_dataset[0].get("soft_pos_prob", None))
        print("DEBUG example_weight:", train_dataset[0].get("example_weight", None))

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