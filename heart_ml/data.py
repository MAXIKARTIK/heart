"""Dataset loading and target extraction."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CDC heart-disease CSV."""
    path = Path(path) if path is not None else config.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Set HEART_DATA_PATH or place "
            "heart_2020_cleaned.csv at the project root."
        )
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where X holds the raw feature columns and y is 1/0.

    The target ``HeartDisease`` is a Yes/No string in the source data.
    """
    if config.TARGET not in df.columns:
        raise KeyError(f"Target column '{config.TARGET}' missing from dataframe")

    y = df[config.TARGET].map({"Yes": 1, "No": 0})
    if y.isna().any():
        bad = df.loc[y.isna(), config.TARGET].unique().tolist()
        raise ValueError(f"Unexpected target values: {bad}")
    y = y.astype("int64")

    X = df[config.RAW_FEATURES].copy()
    return X, y
