"""ORM models."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from .db import Base


class Prediction(Base):
    """Audit log of every risk score served by the API."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    model_name: Mapped[str] = mapped_column(String(64))
    threshold: Mapped[float] = mapped_column(Float)
    probability: Mapped[float] = mapped_column(Float)
    prediction: Mapped[int] = mapped_column(Integer)
    risk_label: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(16), default="single")
    # Raw clinical inputs, stored for auditability / future retraining.
    features: Mapped[dict] = mapped_column(JSON)
