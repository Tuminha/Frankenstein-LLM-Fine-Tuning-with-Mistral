"""Dataset helpers: load CSV, clean, convert to HF datasets, push to Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict


def load_and_clean(csv_path: str):
    """Return cleaned DataFrame with 'text' column.
    
    # TODO: implement trimming, dropna, drop_dupes
    # Hints:
    #   - Load CSV with pandas
    #   - Strip whitespace from 'text' column
    #   - Drop rows with empty text
    #   - Remove duplicates
    # Acceptance:
    #   - Returns DataFrame with 'text' column
    #   - No empty strings or duplicates
    """
    raise NotImplementedError


def df_to_dataset(df, val_split: float):
    """Return DatasetDict with train/validation.
    
    # TODO: implement split with fixed seed
    # Hints:
    #   - Convert DataFrame to Dataset
    #   - Use train_test_split with test_size=val_split
    #   - Set seed for reproducibility
    # Acceptance:
    #   - Returns DatasetDict with 'train' and 'validation' splits
    """
    raise NotImplementedError

