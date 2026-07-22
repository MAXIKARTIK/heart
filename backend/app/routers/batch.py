"""Bulk prediction endpoints.

If a broker (Redis) is configured the work is dispatched to a Celery worker and
the client polls ``/batch/{task_id}``. Otherwise the request is scored inline
and the results are returned immediately.
"""
from __future__ import annotations

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..celery_app import celery_app
from ..core.config import settings
from ..core.security import require_api_key
from ..db import get_db
from ..ml import ModelNotLoadedError, service
from ..models import Prediction
from ..schemas import BatchRequest, BatchResponse, BatchStatusResponse
from ..tasks import batch_predict

router = APIRouter(tags=["batch"], dependencies=[Depends(require_api_key)])


@router.post("/batch", response_model=BatchResponse, summary="Score many patients")
def batch(req: BatchRequest, db: Session = Depends(get_db)) -> BatchResponse:
    records = [r.model_dump() for r in req.records]

    if settings.celery_enabled:
        task = batch_predict.delay(records)
        return BatchResponse(task_id=task.id, status="queued", count=len(records))

    # Synchronous fallback: score + log inline.
    try:
        results = service.predict_many(records)
        meta = service.info()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    for rec, res in zip(records, results):
        db.add(
            Prediction(
                model_name=meta["model_name"],
                threshold=meta["threshold"],
                probability=res["probability"],
                prediction=res["prediction"],
                risk_label=res["risk_label"],
                source="batch",
                features=rec,
            )
        )
    db.commit()
    return BatchResponse(status="completed", count=len(records), results=results)


@router.get("/batch/{task_id}", response_model=BatchStatusResponse, summary="Poll a batch job")
def batch_status(task_id: str) -> BatchStatusResponse:
    if not settings.celery_enabled:
        raise HTTPException(
            status_code=404,
            detail="Async batch is disabled; /batch returns results synchronously.",
        )
    res = AsyncResult(task_id, app=celery_app)
    results = res.result if res.successful() else None
    return BatchStatusResponse(task_id=task_id, status=res.status, results=results)
