"""Pydantic request/response schemas.

The categorical ``Literal`` types mirror ``heart_ml.config.CATEGORY_VALUES``
(the exact values found in the CDC dataset). A consistency check at import time
guarantees the API and the training pipeline never silently drift apart.
"""
from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from heart_ml import config as ml_config

YesNo = Literal["No", "Yes"]


class ClinicalFeatures(BaseModel):
    """The 17 clinical inputs required to score heart-disease risk."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "BMI": 28.5,
                "Smoking": "Yes",
                "AlcoholDrinking": "No",
                "Stroke": "No",
                "PhysicalHealth": 5.0,
                "MentalHealth": 3.0,
                "DiffWalking": "No",
                "Sex": "Male",
                "AgeCategory": "60-64",
                "Race": "White",
                "Diabetic": "Yes",
                "PhysicalActivity": "Yes",
                "GenHealth": "Fair",
                "SleepTime": 6.0,
                "Asthma": "No",
                "KidneyDisease": "No",
                "SkinCancer": "No",
            }
        }
    )

    BMI: float = Field(ge=10, le=100, description="Body Mass Index")
    Smoking: YesNo
    AlcoholDrinking: YesNo
    Stroke: YesNo
    PhysicalHealth: float = Field(ge=0, le=30, description="Unwell physical days (last 30)")
    MentalHealth: float = Field(ge=0, le=30, description="Unwell mental days (last 30)")
    DiffWalking: YesNo
    Sex: Literal["Female", "Male"]
    AgeCategory: Literal[
        "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
        "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older",
    ]
    Race: Literal[
        "White", "Black", "Asian", "American Indian/Alaskan Native",
        "Hispanic", "Other",
    ]
    Diabetic: Literal["No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)"]
    PhysicalActivity: YesNo
    GenHealth: Literal["Poor", "Fair", "Good", "Very good", "Excellent"]
    SleepTime: float = Field(ge=0, le=24, description="Average hours of sleep per 24h")
    Asthma: YesNo
    KidneyDisease: YesNo
    SkinCancer: YesNo


class PredictionResponse(BaseModel):
    id: int | None = Field(default=None, description="DB id of the logged prediction")
    prediction: int = Field(description="1 = predicted heart-disease risk, 0 = not")
    risk_label: str
    probability: float = Field(description="Model probability of the positive class")
    threshold: float = Field(description="Decision threshold applied (tuned for F1)")
    model_name: str
    trained_at: str


class BatchRequest(BaseModel):
    records: list[ClinicalFeatures] = Field(min_length=1, max_length=5000)


class BatchItem(BaseModel):
    prediction: int
    risk_label: str
    probability: float


class BatchResponse(BaseModel):
    task_id: str | None = None
    status: str = Field(description="queued | completed")
    count: int
    results: list[BatchItem] | None = None


class BatchStatusResponse(BaseModel):
    task_id: str
    status: str
    results: list[BatchItem] | None = None


class ModelInfo(BaseModel):
    model_name: str
    threshold: float
    trained_at: str
    sklearn_version: str
    n_train: int
    positive_rate: float
    metrics: dict


# --------------------------------------------------------------------------- #
# Import-time guard: keep the API Literals in sync with the training config.
# --------------------------------------------------------------------------- #
def _assert_categories_in_sync() -> None:
    fields = ClinicalFeatures.model_fields
    for name, allowed in ml_config.CATEGORY_VALUES.items():
        literal_values = set(get_args(fields[name].annotation))
        if literal_values != set(allowed):
            raise RuntimeError(
                f"Schema/config drift for '{name}': schema={literal_values} "
                f"config={set(allowed)}"
            )


_assert_categories_in_sync()
