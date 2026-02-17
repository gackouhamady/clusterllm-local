"""
Robust data preparation for ClusterLLM datasets (subset of 10).

Outputs:
  data/raw/<name>_large.csv
  data/raw/<name>_small.csv
Each CSV: columns = text,label

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
"""

from __future__ import annotations

import os
import re
import gzip
import logging
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

SEED = 42
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/raw")
HF_TOKEN = os.getenv("HF_TOKEN", None)
ALLOW_REMOTE_CODE = os.getenv("ALLOW_REMOTE_CODE", "0") == "1"
STRICT = os.getenv("STRICT", "1") == "1"

# Note: added ".json.gz" for MASSIVE-like repos
SUPPORTED_EXTS = (".jsonl.gz", ".jsonl", ".parquet", ".csv", ".tsv", ".json", ".json.gz")


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _shuffle_df(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= 1:
        return df.reset_index(drop=True)
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def _sample_df(df: pd.DataFrame, n: Optional[int]) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=SEED).reset_index(drop=True)


def _ensure_text_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Missing required columns text/label. cols={df.columns.tolist()}")
    df["text"] = df["text"].astype(str)
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
    # jsonl/jsonl.gz
    if repo_path.endswith(".jsonl.gz"):
        with gzip.open(local_path, "rt", encoding="utf-8") as f:
            return pd.read_json(f, lines=True)
    if repo_path.endswith(".jsonl"):
        return pd.read_json(local_path, lines=True)

    # json.gz (MASSIVE-style)
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


def load_from_hf_files(hf_id: str, split: str, config: Optional[str] = None) -> pd.DataFrame:
    if hf_hub_download is None or list_repo_files is None:
        raise RuntimeError("huggingface_hub is required for file-based loading but is not installed.")

    repo_files = list_repo_files(hf_id, repo_type="dataset", token=HF_TOKEN)
    chosen = _pick_split_files(repo_files, split=split, config=config)
    if not chosen:
        raise RuntimeError(
            f"No supported data files found for {hf_id} split={split} config={config}. "
            f"Repo files sample={repo_files[:80]}"
        )

    parts: List[pd.DataFrame] = []
    for repo_path in chosen:
        local = hf_hub_download(
            repo_id=hf_id,
            filename=repo_path,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        parts.append(_read_one_file(local, repo_path))

    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


def load_from_datasets(
    hf_id: str,
    split: str,
    config: Optional[str] = None,
    allow_remote_code: bool = False,
) -> pd.DataFrame:
    if load_dataset is None:
        raise RuntimeError("datasets is not installed (needed for datasets fallback).")

    kwargs: Dict[str, Any] = {}
    if allow_remote_code:
        if not ALLOW_REMOTE_CODE:
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
    """
    Safe filtering:
      - Try strict: single-label only + remove neutral (27)
      - If empty: relax to only remove neutral (if possible)
      - If still empty: return original df (no filtering)
    """
    out = df.copy()

    # strict
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

        # relaxed: keep multi-label but remove rows that contain neutral if possible
        relaxed = out.copy()
        # if labels is list, remove rows that include 27
        relaxed = relaxed[~relaxed["labels"].apply(lambda x: isinstance(x, list) and 27 in x)]
        # create a stable label: pick first label if list, else None
        relaxed["label"] = relaxed["labels"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        relaxed = relaxed[relaxed["label"].notna()]
        relaxed = relaxed.drop(columns=["labels"], errors="ignore")
        if len(relaxed) > 0:
            return relaxed

        return out  # last resort

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


def save_pair(name: str, large_df: pd.DataFrame, small_df: pd.DataFrame) -> None:
    large_path = os.path.join(OUTPUT_DIR, f"{name}_large.csv")
    small_path = os.path.join(OUTPUT_DIR, f"{name}_small.csv")
    _atomic_write_csv(_ensure_text_label(large_df), large_path)
    _atomic_write_csv(_ensure_text_label(small_df), small_path)
    logger.info(f"Saved {large_path} ({len(large_df)} rows)")
    logger.info(f"Saved {small_path} ({len(small_df)} rows)")


def build_tasks() -> List[Task]:
    tasks: List[Task] = []

    def load_bank77() -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_train = load_from_datasets("banking77", split="train")
        df_test = load_from_datasets("banking77", split="test")
        train = _normalize(df_train, "text", "label")
        test = _normalize(df_test, "text", "label")
        return _sample_df(train, 10003), _sample_df(test, 3080)

    tasks.append(Task("bank77", 10003, 3080, load_bank77))

    def load_clinc_intent() -> Tuple[pd.DataFrame, pd.DataFrame]:
        hf_id = "clinc/clinc_oos"
        cfg = "plus"
        try:
            df_train = load_from_hf_files(hf_id, "train", config=cfg)
            df_test = load_from_hf_files(hf_id, "test", config=cfg)
        except Exception:
            df_train = load_from_datasets(hf_id, split="train", config=cfg)
            df_test = load_from_datasets(hf_id, split="test", config=cfg)

        df_train = clinc_filter_oos(df_train)
        df_test = clinc_filter_oos(df_test)

        text_col = _find_first_col(df_train, ["text", "sentence", "utterance"]) or "text"
        label_col = _find_first_col(df_train, ["intent", "label"]) or "intent"
        train = _normalize(df_train, text_col, label_col)
        test = _normalize(df_test, text_col, label_col)
        return _sample_df(train, 15000), _sample_df(test, 4500)

    tasks.append(Task("clinc_intent", 15000, 4500, load_clinc_intent))

    def load_mtop_intent() -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_train = load_from_datasets("mteb/mtop_intent", split="train", config="en", allow_remote_code=True)
        df_test = load_from_datasets("mteb/mtop_intent", split="test", config="en", allow_remote_code=True)
        train = _normalize(df_train, "text", "label")
        test = _normalize(df_test, "text", "label")
        return _sample_df(train, 15638), _sample_df(test, 2236)

    tasks.append(Task("mtop_intent", 15638, 2236, load_mtop_intent))

    def load_mtop_domain() -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_train = load_from_datasets("mteb/mtop_intent", split="train", config="en", allow_remote_code=True)
        df_test = load_from_datasets("mteb/mtop_intent", split="test", config="en", allow_remote_code=True)
        train = _normalize(df_train, "text", "domain")
        test = _normalize(df_test, "text", "domain")
        return _sample_df(train, 15667), _sample_df(test, 2235)

    tasks.append(Task("mtop_domain", 15667, 2235, load_mtop_domain))

    # MASSIVE: fallback to datasets (since repo layout != train/test naming)
    def load_massive_intent() -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            df_train = load_from_hf_files("mteb/amazon_massive_intent", "train", config="en")
            df_test = load_from_hf_files("mteb/amazon_massive_intent", "test", config="en")
        except Exception:
            df_train = load_from_datasets("mteb/amazon_massive_intent", split="train", config="en")
            df_test = load_from_datasets("mteb/amazon_massive_intent", split="test", config="en")

        train = _normalize(df_train, "text", "label")
        test = _normalize(df_test, "text", "label")
        return _sample_df(train, 11510), _sample_df(test, 2029)

    tasks.append(Task("massive_intent", 11510, 2029, load_massive_intent))

    def load_massive_domain() -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            df_train = load_from_hf_files("mteb/amazon_massive_intent", "train", config="en")
            df_test = load_from_hf_files("mteb/amazon_massive_intent", "test", config="en")
        except Exception:
            df_train = load_from_datasets("mteb/amazon_massive_intent", split="train", config="en")
            df_test = load_from_datasets("mteb/amazon_massive_intent", split="test", config="en")

        train = _normalize(df_train, "text", "scenario")
        test = _normalize(df_test, "text", "scenario")
        return _sample_df(train, 11514), _sample_df(test, 2030)

    tasks.append(Task("massive_domain", 11514, 2030, load_massive_domain))

    def load_mteb_topic(hf_id: str) -> Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]:
        def _load() -> Tuple[pd.DataFrame, pd.DataFrame]:
            df = load_from_hf_files(hf_id, "test", config=None)
            df = explode_sentences_labels(df)

            if "text" in df.columns and "label" in df.columns:
                out = _normalize(df, "text", "label")
            else:
                text_col = _find_first_col(df, ["text", "sentence", "sentences"]) or "text"
                label_col = _find_first_col(df, ["label", "labels"]) or "label"
                out = _normalize(df, text_col, label_col)

            out = _shuffle_df(out)
            large = _sample_df(out, 50000)
            remaining = out.iloc[len(large):].reset_index(drop=True)
            small = _sample_df(remaining if not remaining.empty else out, 5000)
            return large, small

        return _load

    tasks.append(Task("stackex", 50000, 5000, load_mteb_topic("mteb/stackexchange-clustering")))
    tasks.append(Task("arxiv", 50000, 5000, load_mteb_topic("mteb/arxiv-clustering-p2p")))
    tasks.append(Task("reddit", 50000, 5000, load_mteb_topic("mteb/reddit-clustering")))

    def load_go_emotions() -> Tuple[pd.DataFrame, pd.DataFrame]:
        hf_id = "google-research-datasets/go_emotions"
        cfg = "simplified"

        df_train = load_from_hf_files(hf_id, "train", config=cfg)
        df_test = load_from_hf_files(hf_id, "test", config=cfg)

        df_train = goemotions_filter_safe(df_train)
        df_test = goemotions_filter_safe(df_test)

        # figure out label col after safe transform
        label_col_train = "label" if "label" in df_train.columns else "labels"
        label_col_test = "label" if "label" in df_test.columns else "labels"

        train = _normalize(df_train, "text", label_col_train)
        test = _normalize(df_test, "text", label_col_test)

        # Ensure non-empty; if empty, don't filter at all
        if len(train) == 0 or len(test) == 0:
            logger.warning("GoEmotions became empty after filtering; writing unfiltered data instead.")
            raw_train = load_from_hf_files(hf_id, "train", config=cfg)
            raw_test = load_from_hf_files(hf_id, "test", config=cfg)
            # best-effort normalization
            train = _normalize(raw_train, "text", "labels" if "labels" in raw_train.columns else "label")
            test = _normalize(raw_test, "text", "labels" if "labels" in raw_test.columns else "label")

        return _sample_df(train, 23485), _sample_df(test, 3010)

    tasks.append(Task("go_emotions", 23485, 3010, load_go_emotions))

    return tasks


def main() -> None:
    _mkdir(OUTPUT_DIR)
    tasks = build_tasks()

    failures: List[str] = []
    logger.info(f"Running {len(tasks)} dataset tasks. OUTPUT_DIR={OUTPUT_DIR} STRICT={STRICT}")

    for t in tasks:
        logger.info(f"Processing {t.name} ...")
        try:
            large_df, small_df = t.loader()
            large_df = _sample_df(_ensure_text_label(large_df), t.large_n)
            small_df = _sample_df(_ensure_text_label(small_df), t.small_n)
            save_pair(t.name, large_df, small_df)
        except Exception as e:
            logger.error(f"FAILED {t.name}: {e}")
            failures.append(t.name)

    logger.info(f"Done. Success={len(tasks) - len(failures)}/{len(tasks)}")
    if failures:
        logger.error("Failed datasets: " + ", ".join(failures))
        if STRICT:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
