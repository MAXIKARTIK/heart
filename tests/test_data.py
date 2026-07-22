"""Tests for the Data_Loader (`heart_ml.data`).

This module currently hosts the property-based test for **Property 1**
(target extraction and the raw feature table). Example-based and error-path
unit tests for `load_dataset` / `split_features_target` are added by a
sibling task.
"""
from __future__ import annotations

import pandas as pd
from hypothesis import given, strategies as st

from heart_ml import config
from heart_ml.data import split_features_target
from tests.strategies import clinical_records


@st.composite
def yes_no_frames(draw: st.DrawFn) -> tuple[pd.DataFrame, list[str]]:
    """A DataFrame whose ``HeartDisease`` column is only ``"Yes"``/``"No"``.

    The 17 ``RAW_FEATURES`` columns come from the shared
    :func:`clinical_records` strategy; the target column is an independently
    generated list of ``"Yes"``/``"No"`` labels of the same length (always at
    least one row). Returns the frame together with the label list so the test
    can assert the exact ``Yes -> 1`` / ``No -> 0`` mapping row by row.
    """
    labels = draw(st.lists(st.sampled_from(["Yes", "No"]), min_size=1, max_size=25))
    records = draw(
        st.lists(clinical_records(), min_size=len(labels), max_size=len(labels))
    )
    frame = pd.DataFrame(records, columns=config.RAW_FEATURES)
    frame[config.TARGET] = labels
    return frame, labels


# Feature: heart-disease-prediction, Property 1: split_features_target yields only 1/0 targets and exactly the 17 RAW_FEATURES columns in canonical order
@given(yes_no_frames())
def test_split_features_target_maps_labels_and_yields_raw_feature_table(
    frame_and_labels: tuple[pd.DataFrame, list[str]],
) -> None:
    frame, labels = frame_and_labels

    X, y = split_features_target(frame)

    # y is an integer series whose values are confined to {0, 1}.
    assert pd.api.types.is_integer_dtype(y)
    assert set(y.unique().tolist()) <= {0, 1}

    # Each label maps exactly: "Yes" -> 1, "No" -> 0, position by position.
    expected = [1 if label == "Yes" else 0 for label in labels]
    assert y.tolist() == expected

    # X holds exactly the 17 RAW_FEATURES columns, in canonical order.
    assert list(X.columns) == config.RAW_FEATURES


# --------------------------------------------------------------------------- #
# Unit tests: load success and deterministic error paths (Task 2.2).
#
# These example/edge-case tests complement Property 1 above by pinning the
# concrete success path of ``load_dataset`` and the three deterministic error
# paths of ``split_features_target``.
# --------------------------------------------------------------------------- #
import pytest

from heart_ml.data import load_dataset


def test_load_dataset_reads_csv_into_table(tiny_csv) -> None:
    """A valid CSV loads into a table with the target and raw feature columns.

    **Validates: Requirement 1.1**
    """
    df = load_dataset(str(tiny_csv))

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # The loaded table carries the target column and all 17 raw features.
    assert config.TARGET in df.columns
    for column in config.RAW_FEATURES:
        assert column in df.columns


def test_load_dataset_missing_file_raises_file_not_found(tmp_path) -> None:
    """A missing dataset file raises ``FileNotFoundError`` naming the path.

    **Validates: Requirement 1.2**
    """
    missing = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError) as excinfo:
        load_dataset(str(missing))

    # The error message names the expected path so the operator can fix it.
    assert str(missing) in str(excinfo.value)


def test_split_features_target_unexpected_value_raises_value_error(tiny_dataset) -> None:
    """A non-Yes/No target value raises ``ValueError`` listing the offenders.

    **Validates: Requirement 1.4**
    """
    frame = tiny_dataset.copy()
    # Corrupt one target cell with a value that is neither "Yes" nor "No".
    frame.loc[frame.index[0], config.TARGET] = "Maybe"

    with pytest.raises(ValueError) as excinfo:
        split_features_target(frame)

    # The offending value is surfaced in the error message.
    assert "Maybe" in str(excinfo.value)


def test_split_features_target_missing_column_raises_key_error(tiny_dataset) -> None:
    """An absent ``HeartDisease`` column raises ``KeyError`` naming the column.

    **Validates: Requirement 1.5**
    """
    frame = tiny_dataset.drop(columns=[config.TARGET])

    with pytest.raises(KeyError) as excinfo:
        split_features_target(frame)

    # The missing target column is named in the error.
    assert config.TARGET in str(excinfo.value)
