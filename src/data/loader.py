"""
Bank77 dataset loader.

This module downloads the Bank77 dataset from Hugging Face and exports
the train and test splits as CSV files into the local data directory.

The script is designed to be reproducible and compatible with
DVC-based data versioning workflows.
"""

import os
from typing import Tuple

import pandas as pd
from datasets import load_dataset


def _get_output_paths(output_dir: str) -> Tuple[str, str]:
    """
    Build output file paths for the Bank77 dataset.

    Parameters
    ----------
    output_dir : str
        Directory where CSV files will be stored.

    Returns
    -------
    Tuple[str, str]
        Paths to the train and test CSV files.
    """
    train_path = os.path.join(output_dir, "bank77_train.csv")
    test_path = os.path.join(output_dir, "bank77_test.csv")
    return train_path, test_path


def load_bank77(output_dir: str = "data/raw") -> None:
    """
    Download and save the Bank77 dataset as CSV files.

    The dataset is fetched from Hugging Face using the `datasets` library
    and stored locally to enable experiment reproducibility and
    data versioning with DVC.

    Parameters
    ----------
    output_dir : str, optional
        Directory where the dataset will be saved, by default "data/raw".
    """
    print("📥 Downloading Bank77 dataset...")
    dataset = load_dataset("banking77")

    os.makedirs(output_dir, exist_ok=True)
    train_path, test_path = _get_output_paths(output_dir)

    pd.DataFrame(dataset["train"]).to_csv(train_path, index=False)
    pd.DataFrame(dataset["test"]).to_csv(test_path, index=False)

    print(f"✅ Dataset successfully saved to '{output_dir}'")


if __name__ == "__main__":
    load_bank77()
