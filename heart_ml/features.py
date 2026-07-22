"""Feature engineering transformer.

Implemented as a scikit-learn transformer so it lives *inside* the pipeline and
is therefore applied identically at training and serving time. It is stateless
(nothing is learned in ``fit``), which keeps it robust and trivially picklable.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add clinically-motivated features derived from the raw survey fields.

    Added columns:
      * ``TotalUnhealthyDays`` -- physical + mental unhealthy days in the last
        30 days (0-60). A single burden-of-illness signal.
      * ``SleepDeviation``     -- absolute deviation from ~7h of sleep; both too
        little and too much sleep are cardiovascular risk factors.
      * ``IsObese``            -- BMI >= 30 (WHO obesity threshold), 1/0.
    """

    def fit(self, X, y=None):  # noqa: D401 - stateless
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=config.RAW_FEATURES)
        X = X.copy()

        bmi = pd.to_numeric(X["BMI"], errors="coerce")
        phys = pd.to_numeric(X["PhysicalHealth"], errors="coerce")
        ment = pd.to_numeric(X["MentalHealth"], errors="coerce")
        sleep = pd.to_numeric(X["SleepTime"], errors="coerce")

        X["TotalUnhealthyDays"] = (phys + ment).clip(lower=0, upper=60)
        X["SleepDeviation"] = (sleep - 7.0).abs()
        X["IsObese"] = (bmi >= 30.0).astype("int64")
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(config.RAW_FEATURES)
        return list(base) + list(config.ENGINEERED_FEATURES)
