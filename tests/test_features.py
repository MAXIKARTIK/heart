"""Property tests for the Feature_Engineer derived columns (``heart_ml.features``).

Property 2 from the heart-disease-prediction design states that, for every
input record, the three engineered columns are computed exactly per their
documented formulas: ``TotalUnhealthyDays`` clips the sum of unhealthy days to
[0, 60], ``SleepDeviation`` is the absolute distance of sleep from the 7-hour
set-point, and ``IsObese`` flips on at the WHO obesity threshold (BMI >= 30).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from heart_ml import config
from heart_ml.features import FeatureEngineer

from tests.strategies import boundary_clinical_records, clinical_records

# Exercise both typical interior records and records biased toward the numeric
# edge values (BMI == 30.0 where ``IsObese`` flips, SleepTime == 7.0 where
# ``SleepDeviation`` is 0) so the formulas are tested at and around their
# behavioural boundaries.
_records = st.one_of(clinical_records(), boundary_clinical_records())


# Feature: heart-disease-prediction, Property 2: TotalUnhealthyDays=clip(PH+MH,0,60), SleepDeviation=|SleepTime-7|, IsObese=1 iff BMI>=30
@given(records=st.lists(_records, min_size=1, max_size=8))
def test_feature_engineer_computes_derived_columns(records: list[dict[str, Any]]) -> None:
    """Each engineered value matches its formula for every generated row.

    **Validates: Requirements 2.1, 2.2, 2.3**
    """
    frame = pd.DataFrame(records, columns=config.RAW_FEATURES)

    result = FeatureEngineer().fit(frame).transform(frame)

    # Row count is preserved so the per-row assertions line up with ``records``.
    assert len(result) == len(records)

    total_unhealthy = result["TotalUnhealthyDays"].to_numpy()
    sleep_deviation = result["SleepDeviation"].to_numpy()
    is_obese = result["IsObese"].to_numpy()

    for i, original in enumerate(records):
        bmi = float(original["BMI"])
        phys = float(original["PhysicalHealth"])
        ment = float(original["MentalHealth"])
        sleep = float(original["SleepTime"])

        # Requirement 2.1: TotalUnhealthyDays = clip(PhysicalHealth + MentalHealth, 0, 60).
        expected_total = min(60.0, max(0.0, phys + ment))
        assert total_unhealthy[i] == pytest.approx(expected_total, abs=1e-9)
        assert 0.0 <= total_unhealthy[i] <= 60.0

        # Requirement 2.2: SleepDeviation = |SleepTime - 7.0|, always non-negative.
        expected_deviation = abs(sleep - 7.0)
        assert sleep_deviation[i] == pytest.approx(expected_deviation, abs=1e-9)
        assert sleep_deviation[i] >= 0.0

        # Requirement 2.3: IsObese = 1 exactly when BMI >= 30.0, else 0.
        assert int(is_obese[i]) == (1 if bmi >= 30.0 else 0)
        assert int(is_obese[i]) in (0, 1)


# --------------------------------------------------------------------------- #
# Property 3: output schema is stable and stateless (Task 3.2).
# --------------------------------------------------------------------------- #
_EXPECTED_COLUMNS = list(config.RAW_FEATURES) + list(config.ENGINEERED_FEATURES)


# Feature: heart-disease-prediction, Property 3: output columns are RAW_FEATURES + engineered, and transform is independent of what fit observed
@given(
    record=_records,
    fit_data=st.lists(_records, min_size=1, max_size=8),
)
def test_feature_engineer_output_schema_is_stable_and_stateless(
    record: dict[str, Any], fit_data: list[dict[str, Any]]
) -> None:
    """Output columns are fixed, and a record's transform ignores fit data.

    The Feature_Engineer is a stateless transformer: ``fit`` learns nothing, so
    transforming a given record must yield identical output columns and values
    regardless of what (differing) data the transformer was previously fit on.

    **Validates: Requirements 2.4, 2.5**
    """
    record_frame = pd.DataFrame([record], columns=config.RAW_FEATURES)
    other_frame = pd.DataFrame(fit_data, columns=config.RAW_FEATURES)

    # Fit on the single record vs. fit on unrelated data, then transform the
    # same record through both.
    transformed_self = FeatureEngineer().fit(record_frame).transform(record_frame)
    transformed_other = FeatureEngineer().fit(other_frame).transform(record_frame)

    # Requirement 2.4: output columns are exactly RAW_FEATURES + engineered, in order.
    assert list(transformed_self.columns) == _EXPECTED_COLUMNS
    assert list(transformed_other.columns) == _EXPECTED_COLUMNS

    # Requirement 2.5: transform is independent of what fit observed -- the two
    # transforms of the same record are element-for-element identical.
    pd.testing.assert_frame_equal(transformed_self, transformed_other)
