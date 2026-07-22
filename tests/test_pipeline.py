"""Property tests for the Preprocessor (``heart_ml/pipeline.py``).

Covers correctness Property 4 from the heart-disease-prediction design: an
unseen categorical value must transform without raising and produce an
all-zero one-hot indicator block for the affected field.

The full model pipeline runs ``FeatureEngineer`` immediately before the
``ColumnTransformer`` that ``build_preprocessor`` returns, so the preprocessor
never sees a raw record directly -- it receives the engineered frame whose
columns are ``NUMERIC_FEATURES + ENGINEERED_FEATURES`` (scaled) followed by the
13 ``CATEGORICAL_FEATURES`` (one-hot encoded). This module replicates that
ordering exactly so the test exercises the preprocessor the way the pipeline
does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from heart_ml import config
from heart_ml.features import FeatureEngineer
from heart_ml.pipeline import build_preprocessor

from tests.strategies import clinical_records, out_of_domain_categorical

# Columns the ColumnTransformer scales (numeric + engineered) ahead of the
# one-hot categorical blocks; the categorical indicators start at this offset.
_N_SCALED = len(config.NUMERIC_FEATURES) + len(config.ENGINEERED_FEATURES)


def _engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered columns, mirroring the pipeline step that runs
    immediately before the preprocessor (``FeatureEngineer`` is stateless)."""
    return FeatureEngineer().fit_transform(frame)


def _as_frame(record: dict) -> pd.DataFrame:
    """A single raw record as a one-row frame in canonical ``RAW_FEATURES`` order."""
    return pd.DataFrame([record], columns=config.RAW_FEATURES)


def _training_frame() -> pd.DataFrame:
    """A tiny valid frame covering every allowed category for every field.

    Cycling each categorical field through its allowed values, over as many
    rows as the largest category set, guarantees the encoder learns the full
    known domain -- so any injected out-of-domain string is unambiguously
    *unseen*. Numeric values are arbitrary but within their documented ranges.
    """
    n_rows = max(len(values) for values in config.CATEGORY_VALUES.values())
    data: dict[str, list] = {
        "BMI": [15.0 + i for i in range(n_rows)],
        "PhysicalHealth": [float(i % 31) for i in range(n_rows)],
        "MentalHealth": [float((i * 2) % 31) for i in range(n_rows)],
        "SleepTime": [float(4 + (i % 9)) for i in range(n_rows)],
    }
    for name in config.CATEGORICAL_FEATURES:
        allowed = config.CATEGORY_VALUES[name]
        data[name] = [allowed[i % len(allowed)] for i in range(n_rows)]
    return pd.DataFrame(data, columns=config.RAW_FEATURES)


# Fit the preprocessor exactly once: it is read-only across every example, and
# fitting once keeps the property test cheap. The encoder learns the full known
# category domain from the coverage-guaranteed training frame above.
_PREPROCESSOR = build_preprocessor().fit(_engineer(_training_frame()))
_ENCODER = _PREPROCESSOR.named_transformers_["cat"]


def _block_bounds(field: str) -> tuple[int, int]:
    """Half-open ``[start, end)`` column range of ``field``'s one-hot block.

    Located from the encoder's fitted ``categories_`` metadata: categorical
    indicators follow the ``_N_SCALED`` scaled columns, and each field
    contributes ``len(categories_[i])`` columns in ``CATEGORICAL_FEATURES``
    order.
    """
    field_index = config.CATEGORICAL_FEATURES.index(field)
    start = _N_SCALED + sum(
        len(_ENCODER.categories_[j]) for j in range(field_index)
    )
    end = start + len(_ENCODER.categories_[field_index])
    return start, end


@st.composite
def _record_with_unseen_category(draw: st.DrawFn) -> tuple[dict, str]:
    """A valid clinical record with one categorical field set to an unseen
    value, paired with the name of the corrupted field."""
    record = dict(draw(clinical_records()))
    field = draw(st.sampled_from(config.CATEGORICAL_FEATURES))
    record[field] = draw(out_of_domain_categorical(field))
    return record, field


# Feature: heart-disease-prediction, Property 4: an unseen categorical value transforms without error and yields an all-zero indicator block
@given(_record_with_unseen_category())
def test_unknown_category_encodes_as_all_zero_block(
    case: tuple[dict, str],
) -> None:
    record, field = case

    # The transform itself must not raise on the unseen category
    # (OneHotEncoder(handle_unknown="ignore")); reaching the assertions below
    # proves no error was raised.
    transformed = _PREPROCESSOR.transform(_engineer(_as_frame(record)))

    start, end = _block_bounds(field)
    block = transformed[:, start:end]

    # Cross-check via get_feature_names_out that the located columns really are
    # this field's one-hot indicators before asserting on their values.
    block_names = list(_PREPROCESSOR.get_feature_names_out())[start:end]
    assert block_names, f"located no indicator columns for field {field!r}"
    assert all(
        name.startswith(f"{field}_") for name in block_names
    ), f"located columns are not {field!r} indicators: {block_names}"

    # The unseen value must encode as an all-zero indicator block for the field.
    assert np.count_nonzero(block) == 0, (
        f"{field!r} indicator block is not all-zero for an unseen value: "
        f"names={block_names}, values={block.tolist()}"
    )


# --------------------------------------------------------------------------- #
# Unit tests: preprocessor targeting, leak-free fit, and class weight (Task 4.2).
# --------------------------------------------------------------------------- #
from sklearn.tree import DecisionTreeClassifier

from heart_ml.pipeline import decision_tree_pipeline


def _transformer_columns(preprocessor, name: str) -> list[str]:
    """Return the column list a fitted ColumnTransformer routes to ``name``."""
    for tname, _trans, columns in preprocessor.transformers_:
        if tname == name:
            return list(columns)
    raise AssertionError(f"transformer {name!r} not found")


def test_preprocessor_targets_expected_columns_and_drops_others() -> None:
    """Scaler targets numeric+engineered, encoder the 13 categoricals, rest dropped.

    **Validates: Requirements 3.1, 3.2, 3.5**
    """
    preprocessor = build_preprocessor().fit(_engineer(_training_frame()))

    # 3.1: the StandardScaler is applied to numeric + engineered features.
    scaled = _transformer_columns(preprocessor, "num")
    assert scaled == config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES

    # 3.2: the OneHotEncoder is applied to exactly the 13 categorical features.
    encoded = _transformer_columns(preprocessor, "cat")
    assert encoded == config.CATEGORICAL_FEATURES
    assert len(encoded) == 13

    # 3.5: any other column is dropped (remainder="drop").
    assert preprocessor.remainder == "drop"

    # Concretely: feed a frame carrying an extra unrelated column and confirm
    # it contributes no output columns (the scaled + one-hot widths account for
    # every output column).
    engineered = _engineer(_training_frame())
    engineered["UnrelatedExtra"] = 1.0
    transformed = preprocessor.transform(engineered)

    encoder = preprocessor.named_transformers_["cat"]
    expected_width = len(scaled) + sum(len(c) for c in encoder.categories_)
    assert transformed.shape[1] == expected_width


def test_preprocessor_fit_statistics_come_only_from_training_partition() -> None:
    """The scaler's learned mean/scale reflect only the fitted partition (3.3).

    Fitting on a training partition and then on a deliberately different
    partition yields different learned statistics, and each matches the mean of
    the partition it was fit on -- so a held-out set never influences fitting.

    **Validates: Requirement 3.3**
    """
    base = _engineer(_training_frame())

    # A "training" partition and a disjoint "held-out" partition with clearly
    # different numeric distributions (shift BMI far up in the held-out set).
    train_part = base.copy()
    holdout_part = base.copy()
    holdout_part["BMI"] = holdout_part["BMI"] + 100.0

    fitted_on_train = build_preprocessor().fit(train_part)
    scaler_train = fitted_on_train.named_transformers_["num"]
    bmi_index = (config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES).index("BMI")

    # The learned BMI mean equals the training partition's BMI mean, not the
    # held-out partition's (which is 100 higher).
    assert scaler_train.mean_[bmi_index] == pytest.approx(train_part["BMI"].mean())
    assert scaler_train.mean_[bmi_index] != pytest.approx(holdout_part["BMI"].mean())

    # Fitting on the held-out partition learns a different (shifted) mean,
    # confirming fit statistics depend solely on the data passed to fit.
    fitted_on_holdout = build_preprocessor().fit(holdout_part)
    scaler_holdout = fitted_on_holdout.named_transformers_["num"]
    assert scaler_holdout.mean_[bmi_index] == pytest.approx(holdout_part["BMI"].mean())


def test_decision_tree_pipeline_uses_balanced_class_weight() -> None:
    """The Decision Tree classifier applies balanced class weights.

    **Validates: Requirement 5.1**
    """
    pipeline = decision_tree_pipeline()
    classifier = pipeline.named_steps["clf"]

    assert isinstance(classifier, DecisionTreeClassifier)
    assert classifier.class_weight == "balanced"
