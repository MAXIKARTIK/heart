"""Heart disease risk prediction ML package.

Shared library used by both the training pipeline and the FastAPI service.
Keeping the feature engineering, preprocessing and model-building code in one
importable package guarantees that the exact same transformations are applied
at training time and at serving time (and that the serialized pipeline can be
unpickled by the API process).
"""
from __future__ import annotations

from . import config
from .features import FeatureEngineer

__all__ = ["config", "FeatureEngineer", "__version__"]
__version__ = "0.1.0"
