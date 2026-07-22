"""Liveness and model-metadata endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..core.config import settings
from ..ml import ModelNotLoadedError, service
from ..schemas import ModelInfo

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
def health() -> dict:
    return {
        "status": "ok",
        "environment": settings.environment,
        "model_loaded": service.is_loaded,
        "async_enabled": settings.celery_enabled,
    }


@router.get("/health/model", response_model=ModelInfo, summary="Model metadata")
def model_info() -> ModelInfo:
    try:
        return ModelInfo(**service.info())
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
