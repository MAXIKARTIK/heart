"""Shared Hypothesis strategies for the heart-disease-prediction test suite.

These strategies build *raw* clinical records -- plain ``dict``s keyed by
``heart_ml.config.RAW_FEATURES`` -- and stay in lock-step with the single
source of truth in ``heart_ml.config``: feature names come from
``NUMERIC_FEATURES`` / ``CATEGORICAL_FEATURES`` and every categorical value is
drawn from ``CATEGORY_VALUES`` rather than being hardcoded here. The numeric
ranges mirror the documented Clinical_Features bounds.

Provided strategies
-------------------
* :func:`clinical_records`            -- valid raw records (the common case).
* :func:`boundary_clinical_records`   -- valid records biased toward numeric
  edge values (e.g. ``BMI == 30.0`` where ``IsObese`` flips, ``SleepTime ==
  7.0`` where ``SleepDeviation`` is 0).
* :func:`out_of_domain_records`       -- records that violate exactly one field
  constraint (a numeric value out of range or a categorical value outside its
  allowed set), for the validation-rejection properties.

Field-level building blocks (:func:`numeric_value`, :func:`categorical_value`,
:func:`out_of_range_numeric`, :func:`out_of_domain_categorical`) are exposed so
later suites -- notably ``backend/tests`` -- can target a specific field.

This module contains generators only; it does not define any property tests.
"""
from __future__ import annotations

from typing import Any

from hypothesis import strategies as st

from heart_ml import config

# --------------------------------------------------------------------------- #
# Numeric field ranges (documented, inclusive Clinical_Features bounds).
# --------------------------------------------------------------------------- #
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "BMI": (10.0, 100.0),
    "PhysicalHealth": (0.0, 30.0),
    "MentalHealth": (0.0, 30.0),
    "SleepTime": (0.0, 24.0),
}

# Clinically-meaningful edge values worth exercising explicitly:
#   * BMI 30.0        -- the obesity threshold where ``IsObese`` flips 0 -> 1.
#   * SleepTime 7.0   -- the sleep set-point where ``SleepDeviation`` is 0.
#   * the remaining values are each field's inclusive range endpoints.
NUMERIC_BOUNDARIES: dict[str, list[float]] = {
    "BMI": [10.0, 30.0, 100.0],
    "PhysicalHealth": [0.0, 30.0],
    "MentalHealth": [0.0, 30.0],
    "SleepTime": [0.0, 7.0, 24.0],
}


# --------------------------------------------------------------------------- #
# Field-level strategies
# --------------------------------------------------------------------------- #
def _in_range(field: str) -> st.SearchStrategy[float]:
    lo, hi = NUMERIC_BOUNDS[field]
    return st.floats(
        min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False
    )


def numeric_value(field: str, *, boundary: bool = False) -> st.SearchStrategy[float]:
    """A value for one numeric field, within its documented inclusive range.

    When ``boundary`` is true the strategy mixes in the field's
    clinically-meaningful edge values so tests reliably hit them.
    """
    base = _in_range(field)
    if boundary:
        return st.one_of(st.sampled_from(NUMERIC_BOUNDARIES[field]), base)
    return base


def categorical_value(field: str) -> st.SearchStrategy[str]:
    """A value drawn from ``config.CATEGORY_VALUES`` for a categorical field."""
    return st.sampled_from(config.CATEGORY_VALUES[field])


def out_of_range_numeric(field: str) -> st.SearchStrategy[float]:
    """A finite value strictly outside a numeric field's allowed range."""
    lo, hi = NUMERIC_BOUNDS[field]
    span = hi - lo
    below = st.floats(
        min_value=lo - span - 1.0, max_value=lo,
        exclude_max=True, allow_nan=False, allow_infinity=False,
    )
    above = st.floats(
        min_value=hi, max_value=hi + span + 1.0,
        exclude_min=True, allow_nan=False, allow_infinity=False,
    )
    return st.one_of(below, above)


def out_of_domain_categorical(field: str) -> st.SearchStrategy[str]:
    """A non-empty string that is *not* an allowed value for the field."""
    allowed = set(config.CATEGORY_VALUES[field])
    return st.text(min_size=1, max_size=24).filter(lambda v: v not in allowed)


# --------------------------------------------------------------------------- #
# Record strategies
# --------------------------------------------------------------------------- #
def _record_fields(*, boundary: bool) -> dict[str, st.SearchStrategy[Any]]:
    fields: dict[str, st.SearchStrategy[Any]] = {
        name: numeric_value(name, boundary=boundary)
        for name in config.NUMERIC_FEATURES
    }
    fields.update(
        {name: categorical_value(name) for name in config.CATEGORICAL_FEATURES}
    )
    return fields


def clinical_records() -> st.SearchStrategy[dict[str, Any]]:
    """Valid raw clinical records keyed by the 17 ``RAW_FEATURES``.

    Numeric fields fall within their documented inclusive ranges and every
    categorical field is drawn from ``config.CATEGORY_VALUES``. The produced
    ``dict`` has exactly the ``RAW_FEATURES`` keys and no target column.
    """
    return st.fixed_dictionaries(_record_fields(boundary=False))


def boundary_clinical_records() -> st.SearchStrategy[dict[str, Any]]:
    """Valid records biased toward numeric edge values (e.g. ``BMI == 30.0``)."""
    return st.fixed_dictionaries(_record_fields(boundary=True))


@st.composite
def records_with_out_of_range_numeric(
    draw: st.DrawFn, field: str | None = None
) -> dict[str, Any]:
    """A valid record with exactly one numeric field pushed out of range.

    ``field`` pins which numeric field is corrupted; when omitted one of the
    four numeric fields is chosen by the strategy.
    """
    record = dict(draw(clinical_records()))
    target = field if field is not None else draw(st.sampled_from(config.NUMERIC_FEATURES))
    record[target] = draw(out_of_range_numeric(target))
    return record


@st.composite
def records_with_out_of_domain_categorical(
    draw: st.DrawFn, field: str | None = None
) -> dict[str, Any]:
    """A valid record with exactly one categorical field outside its set.

    ``field`` pins which categorical field is corrupted; when omitted one of
    the 13 categorical fields is chosen by the strategy.
    """
    record = dict(draw(clinical_records()))
    target = field if field is not None else draw(st.sampled_from(config.CATEGORICAL_FEATURES))
    record[target] = draw(out_of_domain_categorical(target))
    return record


def out_of_domain_records() -> st.SearchStrategy[dict[str, Any]]:
    """Records violating exactly one field constraint (numeric or categorical)."""
    return st.one_of(
        records_with_out_of_range_numeric(),
        records_with_out_of_domain_categorical(),
    )
