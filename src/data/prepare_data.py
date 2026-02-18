"""
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


def _shuffle_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if len(df) <= 1:
        return df.reset_index(drop=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _sample_df(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


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


@dataclass(frozen=True)
class Task:
    name: str
    large_n: int
    small_n: int
    loader: Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]


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

    def load_bank77() -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_train = load_from_datasets("banking77", split="train", config=None, allow_remote_code=False,
                                      allow_remote_code_env=allow_remote_code_env)
        df_test = load_from_datasets("banking77", split="test", config=None, allow_remote_code=False,
                                     allow_remote_code_env=allow_remote_code_env)
        train = _normalize(df_train, "text", "label")
        test = _normalize(df_test, "text", "label")
        return _sample_df(train, 10003, seed), _sample_df(test, 3080, seed)

    tasks.append(Task("bank77", 10003, 3080, load_bank77))

    def load_clinc_intent() -> Tuple[pd.DataFrame, pd.DataFrame]:
        hf_id = "clinc/clinc_oos"
        cfg = "plus"
        try:
            df_train = load_from_hf_files(hf_id, "train", config=cfg, hf_token=hf_token)
            df_test = load_from_hf_files(hf_id, "test", config=cfg, hf_token=hf_token)
        except Exception:
            df_train = load_from_datasets(hf_id, split="train", config=cfg, allow_remote_code=False,
                                          allow_remote_code_env=allow_remote_code_env)
            df_test = load_from_datasets(hf_id, split="test", config=cfg, allow_remote_code=False,
                                         allow_remote_code_env=allow_remote_code_env)

        df_train = clinc_filter_oos(df_train)
        df_test = clinc_filter_oos(df_test)

        text_col = _find_first_col(df_train, ["text", "sentence", "utterance"]) or "text"
        label_col = _find_first_col(df_train, ["intent", "label"]) or "intent"
        train = _normalize(df_train, text_col, label_col)
        test = _normalize(df_test, text_col, label_col)
        return _sample_df(train, 15000, seed), _sample_df(test, 4500, seed)

    tasks.append(Task("clinc_intent", 15000, 4500, load_clinc_intent))

    def load_mtop_intent() -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_train = load_from_datasets("mteb/mtop_intent", split="train", config="en", allow_remote_code=True,
                                      allow_remote_code_env=allow_remote_code_env)
        df_test = load_from_datasets("mteb/mtop_intent", split="test", config="en", allow_remote_code=True,
                                     allow_remote_code_env=allow_remote_code_env)
        train = _normalize(df_train, "text", "label")
        test = _normalize(df_test, "text", "label")
        return _sample_df(train, 15638, seed), _sample_df(test, 2236, seed)

    tasks.append(Task("mtop_intent", 15638, 2236, load_mtop_intent))

    def load_mtop_domain() -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_train = load_from_datasets("mteb/mtop_intent", split="train", config="en", allow_remote_code=True,
                                      allow_remote_code_env=allow_remote_code_env)
        df_test = load_from_datasets("mteb/mtop_intent", split="test", config="en", allow_remote_code=True,
                                     allow_remote_code_env=allow_remote_code_env)
        train = _normalize(df_train, "text", "domain")
        test = _normalize(df_test, "text", "domain")
        return _sample_df(train, 15667, seed), _sample_df(test, 2235, seed)

    tasks.append(Task("mtop_domain", 15667, 2235, load_mtop_domain))

    def load_massive_intent() -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            df_train = load_from_hf_files("mteb/amazon_massive_intent", "train", config="en", hf_token=hf_token)
            df_test = load_from_hf_files("mteb/amazon_massive_intent", "test", config="en", hf_token=hf_token)
        except Exception:
            df_train = load_from_datasets("mteb/amazon_massive_intent", split="train", config="en",
                                          allow_remote_code=False, allow_remote_code_env=allow_remote_code_env)
            df_test = load_from_datasets("mteb/amazon_massive_intent", split="test", config="en",
                                         allow_remote_code=False, allow_remote_code_env=allow_remote_code_env)

        train = _normalize(df_train, "text", "label")
        test = _normalize(df_test, "text", "label")
        return _sample_df(train, 11510, seed), _sample_df(test, 2029, seed)

    tasks.append(Task("massive_intent", 11510, 2029, load_massive_intent))

    def load_massive_domain() -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            df_train = load_from_hf_files("mteb/amazon_massive_intent", "train", config="en", hf_token=hf_token)
            df_test = load_from_hf_files("mteb/amazon_massive_intent", "test", config="en", hf_token=hf_token)
        except Exception:
            df_train = load_from_datasets("mteb/amazon_massive_intent", split="train", config="en",
                                          allow_remote_code=False, allow_remote_code_env=allow_remote_code_env)
            df_test = load_from_datasets("mteb/amazon_massive_intent", split="test", config="en",
                                         allow_remote_code=False, allow_remote_code_env=allow_remote_code_env)

        train = _normalize(df_train, "text", "scenario")
        test = _normalize(df_test, "text", "scenario")
        return _sample_df(train, 11514, seed), _sample_df(test, 2030, seed)

    tasks.append(Task("massive_domain", 11514, 2030, load_massive_domain))

    def load_mteb_topic(hf_id: str) -> Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]:
        def _load() -> Tuple[pd.DataFrame, pd.DataFrame]:
            df = load_from_hf_files(hf_id, "test", config=None, hf_token=hf_token)
            df = explode_sentences_labels(df)

            if "text" in df.columns and "label" in df.columns:
                out = _normalize(df, "text", "label")
            else:
                text_col = _find_first_col(df, ["text", "sentence", "sentences"]) or "text"
                label_col = _find_first_col(df, ["label", "labels"]) or "label"
                out = _normalize(df, text_col, label_col)

            out = _shuffle_df(out, seed)
            large = _sample_df(out, 50000, seed)
            remaining = out.iloc[len(large):].reset_index(drop=True)
            small = _sample_df(remaining if not remaining.empty else out, 5000, seed)
            return large, small

        return _load

    tasks.append(Task("stackex", 50000, 5000, load_mteb_topic("mteb/stackexchange-clustering")))
    tasks.append(Task("arxiv", 50000, 5000, load_mteb_topic("mteb/arxiv-clustering-p2p")))
    tasks.append(Task("reddit", 50000, 5000, load_mteb_topic("mteb/reddit-clustering")))

    def load_go_emotions() -> Tuple[pd.DataFrame, pd.DataFrame]:
        hf_id = "google-research-datasets/go_emotions"
        cfg = "simplified"

        df_train = load_from_hf_files(hf_id, "train", config=cfg, hf_token=hf_token)
        df_test = load_from_hf_files(hf_id, "test", config=cfg, hf_token=hf_token)

        df_train = goemotions_filter_safe(df_train)
        df_test = goemotions_filter_safe(df_test)

        label_col_train = "label" if "label" in df_train.columns else "labels"
        label_col_test = "label" if "label" in df_test.columns else "labels"

        train = _normalize(df_train, "text", label_col_train)
        test = _normalize(df_test, "text", label_col_test)

        if len(train) == 0 or len(test) == 0:
            logger.warning("GoEmotions became empty after filtering; writing unfiltered data instead.")
            raw_train = load_from_hf_files(hf_id, "train", config=cfg, hf_token=hf_token)
            raw_test = load_from_hf_files(hf_id, "test", config=cfg, hf_token=hf_token)
            train = _normalize(raw_train, "text", "labels" if "labels" in raw_train.columns else "label")
            test = _normalize(raw_test, "text", "labels" if "labels" in raw_test.columns else "label")

        return _sample_df(train, 23485, seed), _sample_df(test, 3010, seed)

    tasks.append(Task("go_emotions", 23485, 3010, load_go_emotions))

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
            large_df, small_df = t.loader()
            large_df = _sample_df(_ensure_text_label(large_df), t.large_n, args.seed)
            small_df = _sample_df(_ensure_text_label(small_df), t.small_n, args.seed)

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
