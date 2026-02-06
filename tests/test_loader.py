import os
import pandas as pd
from unittest.mock import patch

from src.data.loader import load_bank77


@patch("src.data.loader.load_dataset")
def test_load_bank77_creates_csv_files(mock_load_dataset, tmp_path):
    """
    Test that load_bank77:
    - calls load_dataset
    - creates train and test CSV files
    """

    # --- Fake dataset returned by HuggingFace
    mock_load_dataset.return_value = {
        "train": [{"text": "hello", "label": 0}],
        "test": [{"text": "world", "label": 1}],
    }

    # --- Work inside a temp directory (pytest isolation)
    data_dir = tmp_path / "data" / "raw"
    os.makedirs(data_dir, exist_ok=True)

    # --- Run from temp dir
    cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        load_bank77()
    finally:
        os.chdir(cwd)

    # --- Assertions
    train_file = data_dir / "bank77_train.csv"
    test_file = data_dir / "bank77_test.csv"

    assert train_file.exists()
    assert test_file.exists()

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    assert len(df_train) == 1
    assert len(df_test) == 1
