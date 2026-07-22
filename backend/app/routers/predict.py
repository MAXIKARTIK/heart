"""Single-record prediction endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..core.security import require_api_key
from ..db import get_db
from ..ml import ModelNotLoadedError, service
from ..models import Prediction
from ..schemas import ClinicalFeatures, PredictionResponse

router = APIRouter(tags=["predict"], dependencies=[Depends(require_api_key)])


@router.post("/predict", response_model=PredictionResponse, summary="Score one patient")
def predict(features: ClinicalFeatures, db: Session = Depends(get_db)) -> PredictionResponse:
    payload = features.model_dump()
    try:
        result = service.predict_many([payload])[0]
        meta = service.info()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = Prediction(
        model_name=meta["model_name"],
        threshold=meta["threshold"],
        probability=result["probability"],
        prediction=result["prediction"],
        risk_label=result["risk_label"],
        source="single",
        features=payload,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return PredictionResponse(
        id=row.id,
        model_name=meta["model_name"],
        trained_at=meta["trained_at"],
        threshold=meta["threshold"],
        **result,
    )
