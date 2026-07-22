"""Central configuration: paths and feature metadata.

The category values below are taken directly from the CDC ``heart_2020_cleaned``
dataset. They are the single source of truth used by:
  * the preprocessing pipeline (OneHotEncoder categories are learned from data,
    but these lists drive validation and the UI),
  * the FastAPI Pydantic schemas (request validation),
  * the React frontend (dropdown options).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all overridable via environment variables so the same code runs
# locally, in tests, and inside Docker).
# --------------------------------------------------------------------------- #
PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_DIR.parent

DATA_PATH: Path = Path(
    os.getenv("HEART_DATA_PATH", str(PROJECT_ROOT / "heart_2020_cleaned.csv"))
)
ARTIFACTS_DIR: Path = Path(
    os.getenv("HEART_ARTIFACTS_DIR", str(PACKAGE_DIR / "artifacts"))
)
MODEL_PATH: Path = Path(
    os.getenv("HEART_MODEL_PATH", str(ARTIFACTS_DIR / "model.joblib"))
)
PLOTS_DIR: Path = ARTIFACTS_DIR / "plots"

# --------------------------------------------------------------------------- #
# Column groups
# --------------------------------------------------------------------------- #
TARGET: str = "HeartDisease"

# Continuous / count features that get standardized.
NUMERIC_FEATURES: list[str] = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

# Clinically-motivated features derived in FeatureEngineer (also standardized).
ENGINEERED_FEATURES: list[str] = ["TotalUnhealthyDays", "SleepDeviation", "IsObese"]

# String features that get one-hot encoded.
CATEGORICAL_FEATURES: list[str] = [
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "Sex",
    "AgeCategory",
    "Race",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

# The raw fields a client must supply (engineered features are derived server-side).
RAW_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --------------------------------------------------------------------------- #
# Allowed category values (ordered where ordinal), from the dataset.
# --------------------------------------------------------------------------- #
_YES_NO = ["No", "Yes"]

CATEGORY_VALUES: dict[str, list[str]] = {
    "Smoking": _YES_NO,
    "AlcoholDrinking": _YES_NO,
    "Stroke": _YES_NO,
    "DiffWalking": _YES_NO,
    "Sex": ["Female", "Male"],
    "AgeCategory": [
        "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
        "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older",
    ],
    "Race": [
        "White", "Black", "Asian", "American Indian/Alaskan Native",
        "Hispanic", "Other",
    ],
    "Diabetic": ["No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)"],
    "PhysicalActivity": _YES_NO,
    "GenHealth": ["Poor", "Fair", "Good", "Very good", "Excellent"],
    "Asthma": _YES_NO,
    "KidneyDisease": _YES_NO,
    "SkinCancer": _YES_NO,
}

# --------------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 44
TEST_SIZE: float = 0.2
CV_FOLDS: int = 3

# Cap the training set size (stratified) to keep KNN memory/latency bounded and
# training fast. The test set is always a held-out 20% of the FULL dataset so
# reported metrics reflect the real class distribution. Raise via env var to
# train on more data (Decision Tree scales fine; KNN artifact grows with it).
TRAIN_SAMPLE_SIZE: int = int(os.getenv("HEART_TRAIN_SAMPLE", "40000"))
