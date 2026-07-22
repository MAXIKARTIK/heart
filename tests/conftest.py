"""Shared pytest fixtures for the ``heart_ml`` test suite.

The Hypothesis profile (>= 100 examples) is registered and loaded in the
*root* ``conftest.py``; this module deliberately neither re-registers nor
overrides it. It provides only data fixtures shared by the data-loading and
training tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heart_ml import config


def build_tiny_dataframe(
    n: int = 200, positive_rate: float = 0.3, *, seed: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """Build a small, deterministic, stratified heart-disease-style frame.

    The frame has the ``HeartDisease`` Yes/No target column followed by all 17
    ``RAW_FEATURES`` columns, with numeric values inside their documented
    inclusive ranges and categorical values drawn from
    ``config.CATEGORY_VALUES`` (the single source of truth). Both classes are
    always present.

    The defaults (``n=200``, ``positive_rate=0.3``) are chosen so the frame
    survives everything the training pipeline throws at it: the 20% stratified
    test split, the 3-fold cross-validated grid search, and the internal 20%
    validation split -- including KNN with up to 51 neighbours (each CV
    training fold retains well over 51 rows). The frame stays tiny enough that
    a full training run over it is fast.
    """
    rng = np.random.default_rng(seed)
    n_pos = max(1, round(n * positive_rate))
    n_neg = max(1, n - n_pos)
    target = np.array(["Yes"] * n_pos + ["No"] * n_neg, dtype=object)

    data: dict[str, np.ndarray] = {config.TARGET: target}
    # Numeric features, each within its documented inclusive range.
    data["BMI"] = rng.uniform(12.0, 60.0, size=target.size)
    data["PhysicalHealth"] = rng.integers(0, 31, size=target.size).astype(float)
    data["MentalHealth"] = rng.integers(0, 31, size=target.size).astype(float)
    data["SleepTime"] = rng.integers(3, 13, size=target.size).astype(float)
    # Categorical features drawn from the allowed sets.
    for name in config.CATEGORICAL_FEATURES:
        allowed = np.array(config.CATEGORY_VALUES[name], dtype=object)
        data[name] = rng.choice(allowed, size=target.size)

    frame = pd.DataFrame(data, columns=[config.TARGET, *config.RAW_FEATURES])
    # Shuffle so the two classes are interleaved rather than blocked.
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture
def tiny_dataset() -> pd.DataFrame:
    """A fresh, small, stratified heart-disease DataFrame.

    Function-scoped so tests may mutate their copy freely (e.g. to build the
    missing-column / bad-target error cases) without affecting other tests.
    """
    return build_tiny_dataframe()


@pytest.fixture(scope="session")
def tiny_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Path to the tiny dataset written as a CSV (read-only, session-scoped).

    Suitable for ``heart_ml.data.load_dataset`` and for training runs via
    ``heart_ml.train.main(["--data", str(tiny_csv), "--no-plots"])``.
    """
    path = tmp_path_factory.mktemp("heart_data") / "tiny_heart.csv"
    build_tiny_dataframe().to_csv(path, index=False)
    return path
