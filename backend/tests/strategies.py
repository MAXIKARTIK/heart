"""Hypothesis strategies for the backend (FastAPI) test suite.

This module deliberately *reuses* the raw-record strategies defined once in
``tests.strategies`` (the single source of truth for what a valid clinical
record looks like) and lifts them into the backend's own domain type,
``app.schemas.ClinicalFeatures``. Keeping record generation in one place means
the ``heart_ml`` suite and the backend suite can never drift apart on what a
"valid record" is.

Provided strategies
-------------------
* :func:`clinical_features`          -- valid ``ClinicalFeatures`` instances.
* :func:`boundary_clinical_features` -- valid instances biased to numeric edges
  (e.g. ``BMI == 30.0``), useful for the decision-rule / round-trip properties.

The raw-``dict`` strategies and field-level building blocks from
``tests.strategies`` are re-exported so backend tests that need a plain payload
(key-order permutations, missing-field, out-of-range and out-of-domain
rejection cases) can import everything they need from this one module.
"""
from __future__ import annotations

from hypothesis import strategies as st

from app.schemas import ClinicalFeatures

# Reuse the canonical record strategies rather than redefining them here.
from tests.strategies import (  # noqa: F401  (re-exported for backend tests)
    boundary_clinical_records,
    categorical_value,
    clinical_records,
    numeric_value,
    out_of_domain_categorical,
    out_of_domain_records,
    out_of_range_numeric,
    records_with_out_of_domain_categorical,
    records_with_out_of_range_numeric,
)

__all__ = [
    "clinical_features",
    "boundary_clinical_features",
    "clinical_records",
    "boundary_clinical_records",
    "out_of_domain_records",
    "records_with_out_of_range_numeric",
    "records_with_out_of_domain_categorical",
    "numeric_value",
    "categorical_value",
    "out_of_range_numeric",
    "out_of_domain_categorical",
]


def clinical_features() -> st.SearchStrategy[ClinicalFeatures]:
    """Valid :class:`ClinicalFeatures` built from generated raw records.

    Each drawn ``dict`` (keyed by the 17 ``RAW_FEATURES``) is passed straight
    through Pydantic validation, so every produced instance is guaranteed valid.
    """
    return clinical_records().map(lambda rec: ClinicalFeatures(**rec))


def boundary_clinical_features() -> st.SearchStrategy[ClinicalFeatures]:
    """Valid :class:`ClinicalFeatures` biased toward numeric edge values."""
    return boundary_clinical_records().map(lambda rec: ClinicalFeatures(**rec))
