"""
Data Preparation Script for ClusterLLM.

This module downloads, filters, and standardizes the 14 benchmark datasets
required for the ClusterLLM experiments. It handles loading from Hugging Face,
filtering out-of-scope or neutral labels, and sampling large datasets to
match the paper's specifications.

The output is a set of CSV files stored in 'data/raw', ready for DVC tracking.
"""

import os
import logging
import pandas as pd
from datasets import load_dataset
from typing import Optional, List, Dict, Any

# Configure logging to standard output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = "data/raw"
SEED = 42


def setup_directory(directory: str) -> None:
    """Creates the output directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


def filter_clinc_oos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out-of-scope intents from the CLINC dataset.
    Label 150 corresponds to 'oos' (out-of-scope).
    """
    initial_count = len(df)
    # Ensure intent column is numeric if possible, or handle string 'oos'
    if df['intent'].dtype == 'object':
        df = df[df['intent'] != 'oos']
    else:
        df = df[df['intent'] != 150]
    
    logger.info(
        f"Filtered CLINC OOS: {initial_count} -> {len(df)} rows."
    )
    return df


def filter_go_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters GoEmotions to keep only samples with a single label,
    and removes the 'neutral' label (index 27).
    """
    initial_count = len(df)
    
    # Filter for single-label instances
    df['num_labels'] = df['labels'].apply(len)
    df = df[df['num_labels'] == 1]
    
    # Extract the single label
    df['label'] = df['labels'].apply(lambda x: x[0])
    
    # Filter out neutral (27)
    df = df[df['label'] != 27]
    
    # Cleanup columns
    df = df.drop(columns=['num_labels', 'labels'])
    
    logger.info(
        f"Filtered GoEmotions: {initial_count} -> {len(df)} rows."
    )
    return df


def sample_dataset(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Samples n rows from the dataframe if it exceeds n."""
    if len(df) > n:
        logger.info(f"Sampling dataset from {len(df)} to {n} rows.")
        return df.sample(n=n, random_state=SEED)
    return df


def process_single_dataset(
    name: str,
    hf_id: str,
    split: str,
    config: Optional[str],
    label_col: str,
    text_col: str,
    action: Optional[str]
) -> None:
    """
    Loads and processes a single dataset config.

    Args:
        name: The local filename prefix (e.g., 'bank77').
        hf_id: The Hugging Face dataset ID.
        split: The split to load (train/test).
        config: The dataset configuration name (optional).
        label_col: The name of the column containing labels.
        text_col: The name of the column containing text.
        action: Specific preprocessing action ('filter_oos', etc.).
    """
    logger.info(f"Processing {name} from {hf_id} (split={split})...")
    
    try:
        # Load dataset with trust_remote_code=True for MTEB scripts
        if config:
            ds = load_dataset(
                hf_id, config, split=split, trust_remote_code=True
            )
        else:
            ds = load_dataset(hf_id, split=split, trust_remote_code=True)
        
        df = pd.DataFrame(ds)

        # Standardize Text Column
        if text_col not in df.columns:
            # Fallback search for common text columns
            for col in ['sentence', 'text', 'utterance']:
                if col in df.columns:
                    df = df.rename(columns={col: 'text'})
                    break
        else:
            df = df.rename(columns={text_col: 'text'})

        # Standardize Label Column
        if label_col in df.columns:
            df = df.rename(columns={label_col: 'label'})
        
        # Apply specific filters
        if action == "filter_oos":
            df = filter_clinc_oos(df)
        elif action == "filter_neutral":
            df = filter_go_emotions(df)
        elif action == "sample_50k":
            df = sample_dataset(df, 50000)
        elif action == "sample_40k":
            df = sample_dataset(df, 40320)

        # Ensure we have the minimum required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError(
                f"Columns 'text' or 'label' missing in {name}. "
                f"Available: {df.columns.tolist()}"
            )

        # Keep only relevant columns to reduce file size
        output_df = df[['text', 'label']]
        
        output_path = os.path.join(OUTPUT_DIR, f"{name}_large.csv")
        output_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {output_path} ({len(output_df)} rows)")

    except Exception as e:
        logger.error(f"Failed to process {name}: {e}")


def main() -> None:
    """Main execution function defining dataset configurations."""
    setup_directory(OUTPUT_DIR)

    # Configuration Dictionary
    # Format: local_name: (hf_id, split, config, label_col, text_col, action)
    datasets_config: Dict[str, Any] = {
        # Intent Discovery
        "bank77": (
            "banking77", "train", None, "label", "text", None
        ),
        "clinc_intent": (
            "clinc_oos", "train", "plus", "intent", "text", "filter_oos"
        ),
        "mtop_intent": (
            "mteb/mtop_intent", "train", "en", "label", "text", None
        ),
        "massive_intent": (
            "mteb/amazon_massive_intent", "train", "en", "label", "text", None
        ),

        # Domain Discovery
        "clinc_domain": (
            "clinc_oos", "train", "plus", "intent", "text", "filter_oos"
        ),
        "mtop_domain": (
            "mteb/mtop_intent", "train", "en", "domain", "text", None
        ),
        "massive_domain": (
            "mteb/amazon_massive_intent", "train", "en", "scenario", "text", None
        ),

        # Topic Mining
        "stackex": (
            "mteb/stackexchange-clustering", "test", None, "label", "text", "sample_50k"
        ),
        "arxiv": (
            "mteb/arxiv-clustering-p2p", "train", None, "label", "text", "sample_50k"
        ),
        "reddit": (
            "mteb/reddit-clustering", "train", None, "label", "text", "sample_50k"
        ),

        # Type Discovery
        "few_nerd": (
            "few_nerd", "train", "supervised", "fine_grained_type", "text", None
        ),
        "few_rel": (
            "few_rel", "train", None, "label", "text", "sample_40k"
        ),

        # Emotion
        "go_emotions": (
            "go_emotions", "train", None, "labels", "text", "filter_neutral"
        )
    }

    for name, params in datasets_config.items():
        process_single_dataset(name, *params)


if __name__ == "__main__":
    main()