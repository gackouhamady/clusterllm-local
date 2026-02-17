"""
Data Preparation Script for ClusterLLM.

This module downloads, filters, and standardizes the 14 benchmark datasets
required for the ClusterLLM experiments.

Key goals:
- Produce standardized CSVs with columns: ["text", "label"] in data/raw/
- Be robust to dataset column variations (e.g., sentences/labels, intent/domain/scenario)
- Be explicit about splits (some datasets only provide "test")
- Provide actionable errors when a dataset is script-based and not supported by
  the installed `datasets` version.

Outputs:
- data/raw/<name>_large.csv  (large-scale setting)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable, Tuple

import pandas as pd
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "data/raw"
SEED = 42


# -----------------------------
# Helpers
# -----------------------------

def setup_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _first_existing(col_candidates: Iterable[str], columns: Iterable[str]) -> Optional[str]:
    cols = set(columns)
    for c in col_candidates:
        if c in cols:
            return c
    return None


def _normalize_text_label(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    df = df.rename(columns={text_col: "text", label_col: "label"})
    return df[["text", "label"]]


def filter_clinc_oos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out-of-scope intents from CLINC.
    Some versions encode OOS as 'oos' or intent id 150.
    """
    if "intent" in df.columns:
        if df["intent"].dtype == "object":
            df = df[df["intent"] != "oos"]
        else:
            df = df[df["intent"] != 150]
    elif "label" in df.columns:
        # fallback: some versions might store intent as "label"
        # if we can't reliably identify OOS, keep all and log.
        logger.warning("CLINC: couldn't find 'intent' column; skipping OOS filtering.")
    return df


def filter_go_emotions_single_label_no_neutral(df: pd.DataFrame) -> pd.DataFrame:
    """
    GoEmotions: keep only single-label samples and remove neutral (label 27).
    Different HF configs may expose:
      - 'labels' as List[int]
      - or 'label' as int
    """
    if "labels" in df.columns:
        df = df.copy()
        df["num_labels"] = df["labels"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df = df[df["num_labels"] == 1]
        df["label"] = df["labels"].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else None)
        df = df[df["label"].notna()]
        df = df[df["label"] != 27]
        df = df.drop(columns=["labels", "num_labels"], errors="ignore")
        return df

    if "label" in df.columns:
        # If already single-label, just filter neutral
        df = df[df["label"] != 27]
        return df

    raise ValueError(f"GoEmotions: expected 'labels' or 'label' column, got {df.columns.tolist()}")


def sample_dataset(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) > n:
        return df.sample(n=n, random_state=SEED)
    return df


def safe_load(hf_id: str, *, config: Optional[str], split: str) -> Any:
    """
    Load dataset with `datasets.load_dataset` without trust_remote_code.

    If the dataset requires a script (and your `datasets` install disallows it),
    raise a clear, actionable error.
    """
    try:
        if config is None:
            return load_dataset(hf_id, split=split)
        return load_dataset(hf_id, config, split=split)
    except Exception as e:
        msg = str(e)
        # Common messages when scripts / remote code are blocked
        if "Dataset scripts are no longer supported" in msg or "trust_remote_code" in msg:
            raise RuntimeError(
                f"HF dataset '{hf_id}' appears to be script-based and is not loadable "
                f"with your current `datasets` version.\n\n"
                f"Fix (recommended): pin datasets to a 2.x version in your environment, e.g.\n"
                f"  pip install 'datasets==2.20.0'\n\n"
                f"Then rerun.\n"
                f"Original error: {msg}"
            ) from e
        raise


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_id: str
    split: str
    config: Optional[str]
    text_candidates: Tuple[str, ...]
    label_candidates: Tuple[str, ...]
    action: Optional[str] = None
    sample_n: Optional[int] = None


def process_one(spec: DatasetSpec) -> None:
    logger.info(f"Processing {spec.name} from {spec.hf_id} (config={spec.config}, split={spec.split})...")

    ds = safe_load(spec.hf_id, config=spec.config, split=spec.split)
    df = pd.DataFrame(ds)

    # Dataset-specific shape fixes BEFORE standardization
    # MTEB clustering sets sometimes provide lists: sentences/labels
    if "sentences" in df.columns and "labels" in df.columns and spec.name in {"stackex", "arxiv", "reddit"}:
        # explode list columns into row-wise examples
        # sentences: List[str], labels: List[int]
        df = df.explode(["sentences", "labels"], ignore_index=True)
        df = df.rename(columns={"sentences": "text", "labels": "label"})

    # Identify text/label columns robustly
    text_col = _first_existing(spec.text_candidates, df.columns)
    label_col = _first_existing(spec.label_candidates, df.columns)

    if text_col is None and "text" in df.columns:
        text_col = "text"
    if label_col is None and "label" in df.columns:
        label_col = "label"

    # If still missing, give actionable error
    if text_col is None or label_col is None:
        raise ValueError(
            f"{spec.name}: could not find text/label columns.\n"
            f"Columns: {df.columns.tolist()}\n"
            f"Tried text candidates={spec.text_candidates}, label candidates={spec.label_candidates}"
        )

    # Standardize into (text, label) (but keep original cols for actions if needed)
    df = df.rename(columns={text_col: "text", label_col: "label"})

    # Actions
    if spec.action == "filter_oos":
        df = filter_clinc_oos(df)
    elif spec.action == "filter_goemotions":
        df = filter_go_emotions_single_label_no_neutral(df)

    if spec.sample_n is not None:
        df = sample_dataset(df, spec.sample_n)

    # Final keep
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{spec.name}: after processing, missing text/label. Columns={df.columns.tolist()}")

    out = df[["text", "label"]]
    out_path = os.path.join(OUTPUT_DIR, f"{spec.name}_large.csv")
    out.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(out)} rows)")


def main() -> None:
    setup_directory(OUTPUT_DIR)

    # IMPORTANT:
    # Many MTEB clustering datasets only provide "test".
    # We'll use the large-scale split that is available in your environment.
    specs = [
        # Intent discovery
        DatasetSpec(
            name="bank77",
            hf_id="banking77",
            config=None,
            split="train",
            text_candidates=("text", "sentence", "utterance"),
            label_candidates=("label",),
        ),
        DatasetSpec(
            name="clinc_intent",
            hf_id="clinc_oos",
            config="plus",
            split="train",
            text_candidates=("text", "sentence", "utterance"),
            label_candidates=("intent", "label"),
            action="filter_oos",
        ),
        DatasetSpec(
            name="mtop_intent",
            hf_id="mteb/mtop_intent",
            config="en",
            split="train",
            text_candidates=("text", "sentence", "utterance"),
            label_candidates=("label",),
        ),
        DatasetSpec(
            name="massive_intent",
            hf_id="mteb/amazon_massive_intent",
            config="en",
            split="train",
            text_candidates=("text", "sentence", "utterance"),
            label_candidates=("label",),
        ),

        # Domain discovery
        DatasetSpec(
            name="clinc_domain",
            hf_id="clinc_oos",
            config="plus",
            split="train",
            text_candidates=("text", "sentence", "utterance"),
            label_candidates=("domain", "label"),
            action="filter_oos",
        ),
        DatasetSpec(
            name="mtop_domain",
            hf_id="mteb/mtop_intent",
            config="en",
            split="train",
            text_candidates=("text", "sentence", "utterance"),
            label_candidates=("domain",),
        ),
        DatasetSpec(
            name="massive_domain",
            hf_id="mteb/amazon_massive_intent",
            config="en",
            split="train",
            text_candidates=("text", "sentence", "utterance"),
            label_candidates=("scenario",),
        ),

        # Topic mining (MTEB clustering datasets typically only have "test")
        DatasetSpec(
            name="stackex",
            hf_id="mteb/stackexchange-clustering",
            config=None,
            split="test",
            text_candidates=("text", "sentence", "sentences"),
            label_candidates=("label", "labels"),
            sample_n=50000,
        ),
        DatasetSpec(
            name="arxiv",
            hf_id="mteb/arxiv-clustering-p2p",
            config=None,
            split="test",
            text_candidates=("text", "sentence", "sentences"),
            label_candidates=("label", "labels"),
            sample_n=50000,
        ),
        DatasetSpec(
            name="reddit",
            hf_id="mteb/reddit-clustering",
            config=None,
            split="test",
            text_candidates=("text", "sentence", "sentences"),
            label_candidates=("label", "labels"),
            sample_n=50000,
        ),

        # Type discovery
        # NOTE: these may be script-based depending on HF packaging.
        DatasetSpec(
            name="few_rel",
            hf_id="thunlp/few_rel",
            config=None,
            split="train",
            text_candidates=("text", "sentence"),
            label_candidates=("label",),
            sample_n=40320,
        ),

        # Emotion
        DatasetSpec(
            name="go_emotions",
            hf_id="go_emotions",
            config="simplified",
            split="train",
            text_candidates=("text", "sentence"),
            label_candidates=("labels", "label"),
            action="filter_goemotions",
        ),
    ]

    failures = []
    for spec in specs:
        try:
            process_one(spec)
        except Exception as e:
            logger.error(f"Failed {spec.name}: {e}")
            failures.append((spec.name, str(e)))

    if failures:
        logger.error("Some datasets failed. Summary:")
        for name, err in failures:
            logger.error(f"- {name}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
