"""
Merged splits then sampled paper sizes.

Robust data preparation for ClusterLLM datasets (subset of 10).

Outputs (default):
  data/raw/<name>_large.csv
  data/raw/<name>_small.csv
  src/clusterllm/datasets/<name>/large.jsonl
  src/clusterllm/datasets/<name>/small.jsonl

Each CSV: columns = text,label
Each JSONL line: {"text": "...", "label": ...}

Goals:
- Never crash mid-run; continue and summarize failures at the end.
- Prefer HF Hub file loading when possible; fallback to datasets.load_dataset when needed.
- Deterministic sampling (SEED).
- Handles known edge-cases from your environment:
  - MTEB clustering datasets (jsonl/jsonl.gz)
  - MTOP requires remote code
  - MASSIVE repo layout not matching train/test file naming -> fallback to datasets
  - GoEmotions filtering can empty the dataset -> safe fallback filtering

Env:
  HF_TOKEN=...            recommended (rate limits)
  ALLOW_REMOTE_CODE=1     required for MTOP
  STRICT=1                default 1 (exit 1 if any dataset fails)
  OUTPUT_DIR=data/raw     default data/raw
  DATASETS_DIR=src/clusterllm/datasets  default src/clusterllm/datasets
"""

from __future__ import annotations

import os
import re
import gzip
import json
import logging
import argparse
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception:  # pragma: no cover
    hf_hub_download = None
    list_repo_files = None

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None

try:
    from datasets.utils.info_utils import VerificationMode  # type: ignore
except Exception:  # pragma: no cover
    VerificationMode = None  # type: ignore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_data")

SUPPORTED_EXTS = (".jsonl.gz", ".jsonl", ".parquet", ".csv", ".tsv", ".json", ".json.gz")


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    _mkdir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_jsonl(df: pd.DataFrame, path: str) -> None:
    """
    Strict JSONL: each line contains {"text": ..., "label": ...}
    """
    _mkdir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"

    # ensure strict schema
    df = _ensure_text_label(df)

    def _to_py(x: Any) -> Any:
        """
        Convert anything to JSON-serializable Python types:
        - numpy scalars -> python scalars
        - numpy arrays -> lists (with python scalars)
        - lists/tuples containing numpy types -> converted recursively
        """
        try:
            import numpy as np  # local import
            if isinstance(x, np.ndarray):
                return [_to_py(v) for v in x.tolist()]
            if isinstance(x, np.generic):
                return x.item()
        except Exception:
            pass

        if isinstance(x, (list, tuple)):
            return [_to_py(v) for v in x]

        return x

    with open(tmp, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {
                "text": str(row["text"]),
                "label": _to_py(row["label"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    os.replace(tmp, path)


def _infer_text_label(df: pd.DataFrame, *, task: str) -> pd.DataFrame:
    """
    task in {"relation","entity","event","domain","intent","generic"}
    Try hard to build strict (text,label) dataframe.
    """

    if df is None:
        return pd.DataFrame(columns=["text", "label"])

    if hasattr(df, 'empty') and df.empty:
        return pd.DataFrame(columns=["text", "label"])

    out = df.copy()

    # Common MTEB style: sentences + labels
    if "sentences" in out.columns and "labels" in out.columns:
        out = explode_sentences_labels(out)

    # If we still have list-like text fields, explode safely
    if "text" not in out.columns:
        # pick best text column
        text_col = _find_first_col(out, [
            "text", "sentence", "utterance", "query", "input",
            "sent", "sentence1", "sentence_1", "premise"
        ])
        if text_col is None and "sentences" in out.columns:
            text_col = "sentences"
        if text_col is None:
            # last resort: stringify whole row
            out["text"] = out.apply(lambda r: json.dumps(r.to_dict(), ensure_ascii=False), axis=1)
        else:
            out = out.rename(columns={text_col: "text"})

    # label candidates by task
    label_candidates = {
        "relation": ["relation", "relation_type", "rel", "label", "labels", "y"],
        "entity":   ["entity_type", "type", "fine_type", "ner_tag", "label", "labels", "y"],
        "event":    ["event_type", "type", "event", "label", "labels", "y"],
        "domain":   ["domain", "scenario", "service", "label", "labels", "y"],
        "intent":   ["intent", "label", "labels", "y"],
        "generic":  ["label", "labels", "y", "class", "category"],
    }
    cand = label_candidates.get(task, label_candidates["generic"])
    lab_col = _find_first_col(out, cand)

    if lab_col is None:
        # last resort: create dummy labels (but your pipeline wants labels for --measure)
        out["label"] = 0
    else:
        out = out.rename(columns={lab_col: "label"})

    # If labels is list, try to make it single-label when possible
    if "label" in out.columns:
        def _to_single(x):
            if isinstance(x, list) and len(x) > 0:
                return x[0]
            return x
        out["label"] = out["label"].apply(_to_single)

    return _ensure_text_label(out)


def _load_any_splits(hf_id: str, config: Optional[str], allow_remote_code: bool, allow_remote_code_env: bool) -> Dict[str, pd.DataFrame]:
    """
    Load dataset and return dict split_name -> DataFrame.
    Works whether dataset has train/test/validation or only a single split.
    """
    if load_dataset is None:
        raise RuntimeError("datasets is not installed.")

    kwargs: Dict[str, Any] = {}
    if allow_remote_code:
        if not allow_remote_code_env:
            raise RuntimeError(f"{hf_id}: requires remote code but ALLOW_REMOTE_CODE=1 is not set.")
        kwargs["trust_remote_code"] = True
    if VerificationMode is not None:
        kwargs["verification_mode"] = VerificationMode.NO_CHECKS

    ds = load_dataset(hf_id, config, **kwargs) if config else load_dataset(hf_id, **kwargs)

    # ds can be DatasetDict or Dataset
    if hasattr(ds, "keys"):
        return {k: pd.DataFrame(ds[k]) for k in ds.keys()}
    return {"train": pd.DataFrame(ds)}


def _shuffle_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if len(df) <= 1:
        return df.reset_index(drop=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _ensure_text_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Missing required columns text/label. cols={df.columns.tolist()}")
    df["text"] = df["text"].astype(str)
    # labels must exist because get_embedding.sh uses --measure
    if df["label"].isna().any():
        df = df[df["label"].notna()].reset_index(drop=True)
    return df[["text", "label"]]


def _find_first_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _normalize(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    out = df.rename(columns={text_col: "text", label_col: "label"})
    return _ensure_text_label(out)


def _read_one_file(local_path: str, repo_path: str) -> pd.DataFrame:
    if repo_path.endswith(".jsonl.gz"):
        with gzip.open(local_path, "rt", encoding="utf-8") as f:
            return pd.read_json(f, lines=True)
    if repo_path.endswith(".jsonl"):
        return pd.read_json(local_path, lines=True)

    if repo_path.endswith(".json.gz"):
        with gzip.open(local_path, "rt", encoding="utf-8") as f:
            return pd.read_json(f)

    if repo_path.endswith(".parquet"):
        return pd.read_parquet(local_path)
    if repo_path.endswith(".csv"):
        return pd.read_csv(local_path)
    if repo_path.endswith(".tsv"):
        return pd.read_csv(local_path, sep="\t")
    if repo_path.endswith(".json"):
        return pd.read_json(local_path)

    raise ValueError(f"Unsupported file type: {repo_path}")


def _pick_split_files(files: List[str], split: str, config: Optional[str] = None) -> List[str]:
    def ok_ext(p: str) -> bool:
        return p.endswith(SUPPORTED_EXTS)

    cand = [f for f in files if ok_ext(f)]
    cand_cfg = [f for f in cand if config and (f.startswith(config + "/") or ("/" + config + "/") in f)]

    def split_match(p: str) -> bool:
        base = os.path.basename(p)
        return (
            base == f"{split}.jsonl"
            or base == f"{split}.jsonl.gz"
            or base == f"{split}.parquet"
            or base == f"{split}.csv"
            or base == f"{split}.tsv"
            or base == f"{split}.json"
            or base == f"{split}.json.gz"
            or re.match(rf"^{re.escape(split)}-\d+-of-\d+\.parquet$", base) is not None
        )

    preferred = [f for f in cand_cfg if split_match(f)]
    if preferred:
        shards = [f for f in preferred if re.search(r"-\d+-of-\d+\.parquet$", f)]
        return sorted(shards) if shards else sorted(preferred)

    preferred2 = [f for f in cand if split_match(f)]
    if preferred2:
        shards = [f for f in preferred2 if re.search(r"-\d+-of-\d+\.parquet$", f)]
        return sorted(shards) if shards else sorted(preferred2)

    return []


def load_from_hf_files(hf_id: str, split: str, config: Optional[str], hf_token: Optional[str]) -> pd.DataFrame:
    if hf_hub_download is None or list_repo_files is None:
        raise RuntimeError("huggingface_hub is required for file-based loading but is not installed.")

    repo_files = list_repo_files(hf_id, repo_type="dataset", token=hf_token)
    chosen = _pick_split_files(repo_files, split=split, config=config)
    if not chosen:
        raise RuntimeError(
            f"No supported data files found for {hf_id} split={split} config={config}. "
            f"Repo files sample={repo_files[:80]}"
        )

    parts: List[pd.DataFrame] = []
    for repo_path in chosen:
        local = hf_hub_download(repo_id=hf_id, filename=repo_path, repo_type="dataset", token=hf_token)
        parts.append(_read_one_file(local, repo_path))

    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


def load_from_datasets(
    hf_id: str,
    split: str,
    config: Optional[str],
    allow_remote_code: bool,
    allow_remote_code_env: bool,
) -> pd.DataFrame:
    if load_dataset is None:
        raise RuntimeError("datasets is not installed (needed for datasets fallback).")

    kwargs: Dict[str, Any] = {}
    if allow_remote_code:
        if not allow_remote_code_env:
            raise RuntimeError(f"{hf_id}: requires remote code but ALLOW_REMOTE_CODE=1 is not set.")
        kwargs["trust_remote_code"] = True

    if VerificationMode is not None:
        kwargs["verification_mode"] = VerificationMode.NO_CHECKS

    if config is None:
        ds = load_dataset(hf_id, split=split, **kwargs)
    else:
        ds = load_dataset(hf_id, config, split=split, **kwargs)
    return pd.DataFrame(ds)


def clinc_filter_oos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "intent" in out.columns:
        if out["intent"].dtype == "object":
            out = out[out["intent"] != "oos"]
        else:
            out = out[out["intent"] != 150]
    if "label" in out.columns and out["label"].dtype == "object":
        out = out[out["label"] != "oos"]
    return out


def explode_sentences_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "sentences" in df.columns and "labels" in df.columns:
        df = df.explode(["sentences", "labels"], ignore_index=True)
        df = df.rename(columns={"sentences": "text", "labels": "label"})
    return df


def goemotions_filter_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "labels" in out.columns:
        strict = out.copy()
        strict["num_labels"] = strict["labels"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        strict = strict[strict["num_labels"] == 1]
        strict["label"] = strict["labels"].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else None)
        strict = strict[strict["label"].notna()]
        strict = strict[strict["label"] != 27]
        strict = strict.drop(columns=["labels", "num_labels"], errors="ignore")
        if len(strict) > 0:
            return strict

        relaxed = out.copy()
        relaxed = relaxed[~relaxed["labels"].apply(lambda x: isinstance(x, list) and 27 in x)]
        relaxed["label"] = relaxed["labels"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        relaxed = relaxed[relaxed["label"].notna()]
        relaxed = relaxed.drop(columns=["labels"], errors="ignore")
        if len(relaxed) > 0:
            return relaxed

        return out

    if "label" in out.columns:
        relaxed = out[out["label"] != 27]
        return relaxed if len(relaxed) > 0 else out

    return out


# -------------------------
# NEW helpers (minimal + local changes)
# -------------------------

_CANON_SPLITS = ("train", "test", "validation", "valid", "dev", "val")


def load_and_merge_splits(
    hf_id: str,
    config: Optional[str],
    *,
    hf_token: Optional[str],
    allow_remote_code: bool,
    allow_remote_code_env: bool,
    prefer_hf_files: bool = True,
) -> pd.DataFrame:
    """
    Load all available splits then merge them (concat) into a single DataFrame.
    Prefer HF Hub file loading when possible; fallback to datasets.load_dataset.
    """
    parts: List[pd.DataFrame] = []

    if prefer_hf_files:
        for sp in _CANON_SPLITS:
            try:
                df_sp = load_from_hf_files(hf_id, sp, config=config, hf_token=hf_token)
                if df_sp is not None and len(df_sp) > 0:
                    parts.append(df_sp)
            except Exception:
                continue

    if parts:
        return pd.concat(parts, ignore_index=True)

    # fallback: use datasets to discover real splits (handles weird repos / MASSIVE layout, etc.)
    splits = _load_any_splits(
        hf_id,
        config=config,
        allow_remote_code=allow_remote_code,
        allow_remote_code_env=allow_remote_code_env,
    )
    # stable order (deterministic)
    for sp in sorted(splits.keys()):
        df_sp = splits[sp]
        if df_sp is not None and len(df_sp) > 0:
            parts.append(df_sp)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _sample_two_splits_paper_sizes(
    all_df: pd.DataFrame,
    large_n: int,
    small_n: int,
    seed: int,
    *,
    dataset_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministic sampling:
      - shuffle once
      - large = first large_n
      - small = next small_n (disjoint if possible)
    Fallback: if not enough data, pad with sampling (with replacement only if necessary),
    while logging warnings.
    Always returns EXACT sizes (large_n / small_n) when possible (unless all_df empty).
    """
    all_df = _ensure_text_label(all_df)
    n = len(all_df)

    if n == 0:
        logger.warning(f"{dataset_name}: merged dataset is empty after preprocessing.")
        return all_df.copy(), all_df.copy()

    shuf = _shuffle_df(all_df, seed=seed)

    need_total = large_n + small_n
    if n >= need_total:
        large_df = shuf.iloc[:large_n].reset_index(drop=True)
        small_df = shuf.iloc[large_n:large_n + small_n].reset_index(drop=True)
        return large_df, small_df

    logger.warning(
        f"{dataset_name}: not enough rows after merge/preprocess (have={n}, need={need_total}). "
        "Will pad via deterministic sampling (may introduce duplicates and/or overlap)."
    )

    # Build large first (prefer no-replacement slice, then pad)
    base_large = shuf.iloc[:min(large_n, n)].reset_index(drop=True)
    if len(base_large) < large_n:
        pad = shuf.sample(n=(large_n - len(base_large)), replace=True, random_state=seed + 11).reset_index(drop=True)
        large_df = pd.concat([base_large, pad], ignore_index=True)
    else:
        large_df = base_large

    # For small: prefer disjoint remainder, then pad
    remainder = shuf.iloc[min(large_n, n):].reset_index(drop=True)
    base_small = remainder.iloc[:min(small_n, len(remainder))].reset_index(drop=True)
    if len(base_small) < small_n:
        # try sampling from shuf without replacement if possible, else with replacement
        replace = len(shuf) < (small_n - len(base_small))
        pad = shuf.sample(
            n=(small_n - len(base_small)),
            replace=replace,
            random_state=seed + 23,
        ).reset_index(drop=True)
        small_df = pd.concat([base_small, pad], ignore_index=True)
    else:
        small_df = base_small

    # Ensure exact lengths when n>0
    large_df = large_df.iloc[:large_n].reset_index(drop=True)
    small_df = small_df.iloc[:small_n].reset_index(drop=True)
    return large_df, small_df


@dataclass(frozen=True)
class Task:
    name: str
    large_n: int
    small_n: int
    loader: Callable[[], pd.DataFrame]


def _write_outputs(
    name: str,
    large_df: pd.DataFrame,
    small_df: pd.DataFrame,
    output_dir: str,
    datasets_dir: str,
    split_train: str,
    split_eval: str,
    out_train: Optional[str],
    out_eval: Optional[str],
    jsonl_train: Optional[str],
    jsonl_eval: Optional[str],
) -> None:
    # CSV paths
    large_csv = out_train or os.path.join(output_dir, f"{name}_{split_train}.csv")
    small_csv = out_eval or os.path.join(output_dir, f"{name}_{split_eval}.csv")

    # JSONL paths (must match get_embedding.sh)
    ds_dir = os.path.join(datasets_dir, name)
    large_jsonl = jsonl_train or os.path.join(ds_dir, f"{split_train}.jsonl")
    small_jsonl = jsonl_eval or os.path.join(ds_dir, f"{split_eval}.jsonl")

    # Strict schema
    large_df = _ensure_text_label(large_df)
    small_df = _ensure_text_label(small_df)

    _atomic_write_csv(large_df, large_csv)
    _atomic_write_csv(small_df, small_csv)
    _atomic_write_jsonl(large_df, large_jsonl)
    _atomic_write_jsonl(small_df, small_jsonl)

    logger.info(f"Saved CSV  : {large_csv} ({len(large_df)} rows)")
    logger.info(f"Saved CSV  : {small_csv} ({len(small_df)} rows)")
    logger.info(f"Saved JSONL: {large_jsonl} ({len(large_df)} rows)")
    logger.info(f"Saved JSONL: {small_jsonl} ({len(small_df)} rows)")


def build_tasks(
    seed: int,
    hf_token: Optional[str],
    allow_remote_code_env: bool,
) -> List[Task]:
    tasks: List[Task] = []

    # Paper sizes (Table 1) enforced by Task.large_n/small_n below.

    def load_bank77() -> pd.DataFrame:
        hf_id = "banking77"
        all_df = load_and_merge_splits(
            hf_id, config=None,
            hf_token=hf_token,
            allow_remote_code=False,
            allow_remote_code_env=allow_remote_code_env,
            prefer_hf_files=False,  # simple + robust
        )
        # normalize
        all_df = _normalize(all_df, "text", "label")
        return all_df

    tasks.append(Task("bank77", 10003, 3080, load_bank77))

    def load_clinc_intent() -> pd.DataFrame:
        hf_id = "clinc/clinc_oos"
        cfg = "plus"
        all_df = load_and_merge_splits(
            hf_id, config=cfg,
            hf_token=hf_token,
            allow_remote_code=False,
            allow_remote_code_env=allow_remote_code_env,
            prefer_hf_files=True,
        )
        before = len(all_df)
        all_df = clinc_filter_oos(all_df)
        after = len(all_df)
        text_col = _find_first_col(all_df, ["text", "sentence", "utterance"]) or "text"
        label_col = _find_first_col(all_df, ["intent", "label"]) or "intent"
        out = _normalize(all_df, text_col, label_col)
        logger.info(f"clinc_intent: rows before OOS filter={before}, after={after}, normalized={len(out)}")
        return out

    tasks.append(Task("clinc_intent", 15000, 4500, load_clinc_intent))

    # Keep this dataset name as in your codebase (CLINC(I) in paper)
    def load_clinc150() -> pd.DataFrame:
        hf_id = "clinc/clinc_oos"
        cfg = "plus"
        all_df = load_and_merge_splits(
            hf_id, config=cfg,
            hf_token=hf_token,
            allow_remote_code=False,
            allow_remote_code_env=allow_remote_code_env,
            prefer_hf_files=True,
        )
        before = len(all_df)
        all_df = clinc_filter_oos(all_df)
        after = len(all_df)
        text_col = _find_first_col(all_df, ["text", "sentence", "utterance"]) or "text"
        label_col = _find_first_col(all_df, ["intent", "label"]) or "intent"
        out = _normalize(all_df, text_col, label_col)
        logger.info(f"clinc150: rows before OOS filter={before}, after={after}, normalized={len(out)}")
        return out

    tasks.append(Task("clinc150", 15000, 4500, load_clinc150))

    def load_mtop_intent() -> pd.DataFrame:
        hf_id = "mteb/mtop_intent"
        all_df = load_and_merge_splits(
            hf_id, config="en",
            hf_token=hf_token,
            allow_remote_code=True,
            allow_remote_code_env=allow_remote_code_env,
            prefer_hf_files=False,  # MTOP often via datasets + remote code
        )
        out = _normalize(all_df, "text", "label")
        return out

    # Paper MTOP(I) large=15638 small=4386
    tasks.append(Task("mtop_intent", 15638, 4386, load_mtop_intent))

    def load_mtop_domain() -> pd.DataFrame:
        hf_id = "mteb/mtop_domain"
        all_df = load_and_merge_splits(
            hf_id, config="en",
            hf_token=hf_token,
            allow_remote_code=True,
            allow_remote_code_env=allow_remote_code_env,
            prefer_hf_files=False,
        )
        out = _normalize(all_df, "text", "domain")
        return out

    # Paper MTOP(D) large=15667 small=4386
    tasks.append(Task("mtop_domain", 15667, 4386, load_mtop_domain))

    def load_massive_intent() -> pd.DataFrame:
        hf_id = "mteb/amazon_massive_intent"
        all_df = load_and_merge_splits(
            hf_id, config="en",
            hf_token=hf_token,
            allow_remote_code=False,
            allow_remote_code_env=allow_remote_code_env,
            prefer_hf_files=True,  # try hub files first; fallback handles weird layout
        )
        out = _normalize(all_df, "text", "label")
        return out

    tasks.append(Task("massive_intent", 11510, 2974, load_massive_intent))

    def load_massive_domain() -> pd.DataFrame:
        hf_id = "mteb/amazon_massive_domain"
        all_df = load_and_merge_splits(
            hf_id, config="en",
            hf_token=hf_token,
            allow_remote_code=False,
            allow_remote_code_env=allow_remote_code_env,
            prefer_hf_files=True,
        )
        # MASSIVE domain: scenario as label (paper)
        out = _normalize(all_df, "text", "scenario")
        return out

    tasks.append(Task("massive_domain", 11514, 2974, load_massive_domain))

    def load_mteb_topic(hf_id: str) -> Callable[[], pd.DataFrame]:
        def _load() -> pd.DataFrame:
            all_df = load_and_merge_splits(
                hf_id, config=None,
                hf_token=hf_token,
                allow_remote_code=False,
                allow_remote_code_env=allow_remote_code_env,
                prefer_hf_files=True,
            )
            before = len(all_df)
            all_df = explode_sentences_labels(all_df)
            # robust infer/normalize
            if "text" in all_df.columns and "label" in all_df.columns:
                out = _normalize(all_df, "text", "label")
            else:
                text_col = _find_first_col(all_df, ["text", "sentence", "sentences"]) or "text"
                label_col = _find_first_col(all_df, ["label", "labels"]) or "label"
                out = _normalize(all_df, text_col, label_col)
            logger.info(f"{hf_id}: rows before explode={before}, after explode/normalize={len(out)}")
            return out
        return _load

    tasks.append(Task("stackex", 50000, 4156, load_mteb_topic("mteb/stackexchange-clustering")))
    tasks.append(Task("arxiv", 50000, 3674, load_mteb_topic("mteb/arxiv-clustering-p2p")))
    tasks.append(Task("reddit", 50000, 3217, load_mteb_topic("mteb/reddit-clustering")))

    def load_go_emotions() -> pd.DataFrame:
        hf_id = "google-research-datasets/go_emotions"
        cfg = "simplified"
        all_df = load_and_merge_splits(
            hf_id, config=cfg,
            hf_token=hf_token,
            allow_remote_code=False,
            allow_remote_code_env=allow_remote_code_env,
            prefer_hf_files=True,
        )
        before = len(all_df)
        filtered = goemotions_filter_safe(all_df)
        after = len(filtered)

        label_col = "label" if "label" in filtered.columns else ("labels" if "labels" in filtered.columns else "label")
        out = _normalize(filtered, "text", label_col)

        if len(out) == 0:
            logger.warning("GoEmotions became empty after filtering; writing unfiltered data instead.")
            raw = all_df
            raw_label_col = "labels" if "labels" in raw.columns else ("label" if "label" in raw.columns else "labels")
            out = _normalize(raw, "text", raw_label_col)

        logger.info(f"go_emotions: rows before filter={before}, after filter={after}, normalized={len(out)}")
        return out

    tasks.append(Task("go_emotions", 23485, 5940, load_go_emotions))

    # -------------------------
    # FewRel / FewNerd / FewEvent / CLINC(D)
    # -------------------------

    def load_few_rel_nat() -> pd.DataFrame:
        hf_id = "BrandonZYW/FewRelClustering"
        all_df = load_and_merge_splits(
            hf_id, config=None,
            hf_token=hf_token,
            allow_remote_code=False,
            allow_remote_code_env=allow_remote_code_env,
            prefer_hf_files=False,  # these often work best via datasets split discovery
        )
        before = len(all_df)
        if "cluster" in all_df.columns:
            all_df = all_df.rename(columns={"cluster": "label"})
        out = _ensure_text_label(all_df)
        logger.info(f"few_rel_nat: rows merged={before}, normalized={len(out)}")
        return out

    tasks.append(Task("few_rel_nat", 40320, 4480, load_few_rel_nat))

    def load_few_nerd_nat() -> pd.DataFrame:
        hf_id = "BrandonZYW/FewNerdClustering"
        all_df = load_and_merge_splits(
            hf_id, config=None,
            hf_token=hf_token,
            allow_remote_code=False,
            allow_remote_code_env=allow_remote_code_env,
            prefer_hf_files=False,
        )
        before = len(all_df)
        if "cluster" in all_df.columns:
            all_df = all_df.rename(columns={"cluster": "label"})
        out = _ensure_text_label(all_df)
        logger.info(f"few_nerd_nat: rows merged={before}, normalized={len(out)}")
        return out

    tasks.append(Task("few_nerd_nat", 50000, 3789, load_few_nerd_nat))

    def load_few_event() -> pd.DataFrame:
        hf_id = "BrandonZYW/FewEventClustering"
        all_df = load_and_merge_splits(
            hf_id, config=None,
            hf_token=hf_token,
            allow_remote_code=False,
            allow_remote_code_env=allow_remote_code_env,
            prefer_hf_files=False,
        )
        before = len(all_df)
        if "cluster" in all_df.columns:
            all_df = all_df.rename(columns={"cluster": "label"})
        out = _ensure_text_label(all_df)
        logger.info(f"few_event: rows merged={before}, normalized={len(out)}")
        return out

    tasks.append(Task("few_event", 18969, 4742, load_few_event))

    def load_clinc_domain() -> pd.DataFrame:
        hf_id = "willcb/clinc-domain"
        all_df = load_and_merge_splits(
            hf_id, config=None,
            hf_token=hf_token,
            allow_remote_code=False,
            allow_remote_code_env=allow_remote_code_env,
            prefer_hf_files=False,
        )
        before = len(all_df)
        if "input" in all_df.columns and "text" not in all_df.columns:
            all_df = all_df.rename(columns={"input": "text"})
        out = _ensure_text_label(all_df)
        logger.info(f"clinc_domain: rows merged={before}, normalized={len(out)}")
        return out

    # Paper CLINC(D): large=15000 small=4500
    tasks.append(Task("clinc_domain", 15000, 4500, load_clinc_domain))

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, help="If set, prepare only this dataset name.")
    parser.add_argument("--seed", default=int(os.getenv("SEED", "42")), type=int)
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "data/raw"), type=str)
    parser.add_argument("--datasets-dir", default=os.getenv("DATASETS_DIR", "src/clusterllm/datasets"), type=str)
    parser.add_argument("--split-train", default="large", type=str)
    parser.add_argument("--split-eval", default="small", type=str)

    # Optional explicit output paths (useful for DVC)
    parser.add_argument("--out-train", default=None, type=str)
    parser.add_argument("--out-eval", default=None, type=str)
    parser.add_argument("--jsonl-train", default=None, type=str)
    parser.add_argument("--jsonl-eval", default=None, type=str)

    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN", None)
    allow_remote_code_env = os.getenv("ALLOW_REMOTE_CODE", "0") == "1"
    strict = os.getenv("STRICT", "1") == "1"

    _mkdir(args.output_dir)
    _mkdir(args.datasets_dir)

    tasks = build_tasks(seed=args.seed, hf_token=hf_token, allow_remote_code_env=allow_remote_code_env)

    failures: List[str] = []
    logger.info(f"Running {len(tasks)} dataset tasks. OUTPUT_DIR={args.output_dir} STRICT={strict}")

    for t in tasks:
        if args.dataset and t.name != args.dataset:
            continue

        logger.info(f"Processing {t.name} ...")
        try:
            all_df = t.loader()

            # log merged/preprocessed count
            try:
                all_df_checked = _ensure_text_label(all_df)
            except Exception as e:
                raise RuntimeError(f"{t.name}: after preprocessing, text/label not valid: {e}")

            logger.info(f"{t.name}: merged+preprocessed rows={len(all_df_checked)}")

            # Deterministic disjoint sampling (paper sizes)
            large_df, small_df = _sample_two_splits_paper_sizes(
                all_df_checked, t.large_n, t.small_n, args.seed, dataset_name=t.name
            )

            logger.info(
                f"{t.name}: final sizes -> large={len(large_df)}/{t.large_n}, small={len(small_df)}/{t.small_n}"
            )

            # In single-dataset mode, allow explicit paths
            out_train = args.out_train if args.dataset else None
            out_eval = args.out_eval if args.dataset else None
            jsonl_train = args.jsonl_train if args.dataset else None
            jsonl_eval = args.jsonl_eval if args.dataset else None

            _write_outputs(
                name=t.name,
                large_df=large_df,
                small_df=small_df,
                output_dir=args.output_dir,
                datasets_dir=args.datasets_dir,
                split_train=args.split_train,
                split_eval=args.split_eval,
                out_train=out_train,
                out_eval=out_eval,
                jsonl_train=jsonl_train,
                jsonl_eval=jsonl_eval,
            )
        except Exception as e:
            logger.error(f"FAILED {t.name}: {e}")
            failures.append(t.name)

    logger.info(f"Done. Success={len(tasks) - len(failures)}/{len(tasks)}")
    if failures:
        logger.error("Failed datasets: " + ", ".join(failures))
        if strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()