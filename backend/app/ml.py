"""Model-serving layer.

Loads the serialized pipeline bundle produced by ``heart_ml.train`` and exposes
thread-safe single/batch prediction. The pipeline itself performs feature
engineering + preprocessing, so the service only has to hand it a DataFrame of
the raw clinical fields.
"""
from __future__ import annotations

import threading
from pathlib import Path

import joblib
import pandas as pd

from heart_ml import config as ml_config

from .core.config import settings


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted but no model artifact is available."""


class PredictionService:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model_path = Path(model_path or settings.model_path or ml_config.MODEL_PATH)
        self._bundle: dict | None = None
        self._lock = threading.Lock()

    @property
    def model_path(self) -> Path:
        return self._model_path

    def load(self, force: bool = False) -> dict:
        if self._bundle is not None and not force:
            return self._bundle
        with self._lock:
            if self._bundle is None or force:
                if not self._model_path.exists():
                    raise ModelNotLoadedError(
                        f"Model artifact not found at {self._model_path}. "
                        "Train it first: `python -m heart_ml.train`."
                    )
                self._bundle = joblib.load(self._model_path)
        return self._bundle

    @property
    def is_loaded(self) -> bool:
        return self._bundle is not None

    def info(self) -> dict:
        b = self.load()
        return {
            "model_name": b["model_name"],
            "threshold": float(b["threshold"]),
            "trained_at": b["trained_at"],
            "sklearn_version": b["sklearn_version"],
            "n_train": int(b["n_train"]),
            "positive_rate": float(b["positive_rate"]),
            "metrics": b["metrics"],
        }

    def predict_many(self, records: list[dict]) -> list[dict]:
        """Score a list of raw-feature dicts. Returns per-record results."""
        b = self.load()
        # Enforce column order expected by the pipeline.
        X = pd.DataFrame(records)[ml_config.RAW_FEATURES]
        proba = b["pipeline"].predict_proba(X)[:, 1]
        threshold = float(b["threshold"])
        results = []
        for p in proba:
            p = float(p)
            pred = int(p >= threshold)
            results.append(
                {
                    "prediction": pred,
                    "probability": p,
                    "risk_label": "At risk" if pred else "Low risk",
                }
            )
        return results


# Module-level singleton reused across requests and Celery tasks.
service = PredictionService()
