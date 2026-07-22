"""Property + unit tests for the Validation_Layer (``backend/app/schemas.py``).

Validates the request-validation correctness properties from the design:

* Property 14 -- a validated ``ClinicalFeatures`` survives a JSON round-trip
  (task 10.1).
* Property 15 -- dropping any one of the 17 required fields is rejected
  (task 10.2).
* Property 16 -- out-of-range numeric values are rejected, in-range accepted
  (task 10.3).
* Property 17 -- out-of-domain categorical values are rejected (task 10.4).
* Property 18 -- ``BatchRequest`` enforces the 1..5000 size bounds (task 10.5).

plus unit tests for the schema/config consistency guard ``_assert_categories_in_sync``
(task 10.6): it passes when the schema matches ``heart_ml.config.CATEGORY_VALUES``
and raises a drift ``RuntimeError`` naming the field and both value sets when a
category is perturbed.

Each property test is a single Hypothesis test running >= 100 examples.
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

# The batch-size tests reuse one valid record (read-only) across generated
# sizes; only the record count varies, so reusing the fixture is safe.
_READ_ONLY_FIXTURE = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)

from heart_ml import config as ml_config

import app.schemas as schemas_module
from app.schemas import BatchRequest, ClinicalFeatures

from tests.strategies import (
    clinical_records,
    records_with_out_of_domain_categorical,
    records_with_out_of_range_numeric,
)
from tests.strategies import out_of_range_numeric as _oor_numeric


# --------------------------------------------------------------------------- #
# Property 14: validated requests survive a JSON round-trip (task 10.1).
# --------------------------------------------------------------------------- #
# Feature: heart-disease-prediction, Property 14: serializing a valid ClinicalFeatures to JSON and re-parsing yields an equivalent object
@given(record=clinical_records())
def test_validated_request_survives_json_round_trip(record) -> None:
    """A valid ``ClinicalFeatures`` re-parses from its own JSON unchanged.

    **Validates: Property 14; Requirements 10.6**
    """
    original = ClinicalFeatures(**record)
    restored = ClinicalFeatures.model_validate_json(original.model_dump_json())
    assert restored == original


# --------------------------------------------------------------------------- #
# Property 15: missing required fields are rejected (task 10.2).
# --------------------------------------------------------------------------- #
# Feature: heart-disease-prediction, Property 15: dropping any one of the 17 required fields causes a validation error
@given(record=clinical_records(), drop=st.sampled_from(ml_config.RAW_FEATURES))
def test_missing_required_field_is_rejected(record, drop: str) -> None:
    """Dropping any single one of the 17 required fields fails validation.

    **Validates: Property 15; Requirements 10.1**
    """
    payload = dict(record)
    del payload[drop]

    with pytest.raises(ValidationError) as excinfo:
        ClinicalFeatures(**payload)
    # The error must implicate the dropped field.
    assert any(err["loc"] == (drop,) for err in excinfo.value.errors())


# --------------------------------------------------------------------------- #
# Property 16: out-of-range numeric values rejected; in-range accepted (task 10.3).
# --------------------------------------------------------------------------- #
# Feature: heart-disease-prediction, Property 16: BMI outside [10,100] / PhysicalHealth,MentalHealth outside [0,30] / SleepTime outside [0,24] are rejected; in-range accepted
@given(
    record=clinical_records(),
    field=st.sampled_from(ml_config.NUMERIC_FEATURES),
    data=st.data(),
)
def test_out_of_range_numeric_rejected_in_range_accepted(record, field: str, data) -> None:
    """Numeric fields are rejected out of range and accepted within range.

    **Validates: Property 16; Requirements 10.2, 10.3, 10.4**
    """
    # In-range payload validates.
    ClinicalFeatures(**record)

    # Push exactly one numeric field out of its documented range -> rejected.
    bad = dict(record)
    bad[field] = data.draw(_oor_numeric(field))
    with pytest.raises(ValidationError) as excinfo:
        ClinicalFeatures(**bad)
    assert any(err["loc"] == (field,) for err in excinfo.value.errors())


# --------------------------------------------------------------------------- #
# Property 17: out-of-domain categorical values are rejected (task 10.4).
# --------------------------------------------------------------------------- #
# Feature: heart-disease-prediction, Property 17: any categorical value outside its allowed set is rejected
@given(record=records_with_out_of_domain_categorical())
def test_out_of_domain_categorical_rejected(record) -> None:
    """A categorical value outside its allowed set fails validation.

    **Validates: Property 17; Requirements 10.5**
    """
    with pytest.raises(ValidationError):
        ClinicalFeatures(**record)


# --------------------------------------------------------------------------- #
# Property 18: batch size bounds are enforced 1..5000 (task 10.5).
# --------------------------------------------------------------------------- #
# Feature: heart-disease-prediction, Property 18: BatchRequest accepts 1..5000 records, rejects 0 or >5000
@_READ_ONLY_FIXTURE
@given(
    size=st.one_of(
        st.just(0),                                    # below the lower bound
        st.integers(min_value=1, max_value=5000),      # accepted range (incl. bounds)
        st.integers(min_value=5001, max_value=5200),   # above the upper bound
    ),
)
def test_batch_size_bounds_are_enforced(size: int, sample_clinical_record: dict) -> None:
    """``BatchRequest`` accepts 1..5000 records inclusive and rejects 0 or >5000.

    The single valid record is replicated ``size`` times; only the record
    *count* is under test, so replication keeps generation cheap while still
    exercising the boundary sizes (0, 1, 5000, 5001).

    **Validates: Property 18; Requirements 12.1**
    """
    records = [dict(sample_clinical_record) for _ in range(size)]
    if 1 <= size <= 5000:
        batch = BatchRequest(records=records)
        assert len(batch.records) == size
    else:
        with pytest.raises(ValidationError):
            BatchRequest(records=records)


# --------------------------------------------------------------------------- #
# Unit tests: schema/config consistency guard (task 10.6).
# --------------------------------------------------------------------------- #
def test_assert_categories_in_sync_passes_when_schema_matches_config() -> None:
    """The guard is a no-op when the schema matches ``CATEGORY_VALUES``.

    **Validates: Requirement 11.1**
    """
    # The module imported cleanly (the guard runs at import time), and calling
    # it again against the unmodified schema must not raise.
    schemas_module._assert_categories_in_sync()


def test_assert_categories_in_sync_raises_on_drift(monkeypatch) -> None:
    """Perturbing a category makes the guard raise, naming field + both sets.

    **Validates: Requirement 11.2**
    """
    perturbed = {k: list(v) for k, v in ml_config.CATEGORY_VALUES.items()}
    perturbed["Smoking"] = ["No", "Yes", "Sometimes"]  # drift: extra value

    monkeypatch.setattr(schemas_module.ml_config, "CATEGORY_VALUES", perturbed)

    with pytest.raises(RuntimeError) as excinfo:
        schemas_module._assert_categories_in_sync()
    message = str(excinfo.value)
    assert "Smoking" in message          # names the drifting field
    assert "Sometimes" in message        # includes the config value set
