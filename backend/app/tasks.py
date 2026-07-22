"""Celery tasks for asynchronous / bulk scoring."""
from __future__ import annotations

from .celery_app import celery_app
from .db import SessionLocal
from .ml import service
from .models import Prediction


def _persist(records: list[dict], results: list[dict], source: str) -> None:
    """Write scored records to the audit log using a fresh session."""
    meta = service.info()
    db = SessionLocal()
    try:
        for rec, res in zip(records, results):
            db.add(
                Prediction(
                    model_name=meta["model_name"],
                    threshold=meta["threshold"],
                    probability=res["probability"],
                    prediction=res["prediction"],
                    risk_label=res["risk_label"],
                    source=source,
                    features=rec,
                )
            )
        db.commit()
    finally:
        db.close()


@celery_app.task(name="app.tasks.batch_predict")
def batch_predict(records: list[dict]) -> list[dict]:
    """Score many records and log them. Runs on a worker (or inline if eager)."""
    results = service.predict_many(records)
    _persist(records, results, source="batch")
    return results
