"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .core.config import settings
from .db import init_db
from .ml import service
from .routers import batch, health, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables for local/SQLite runs; PostgreSQL deployments use Alembic.
    if settings.database_url.startswith("sqlite"):
        try:
            init_db()
            logger.info("Initialized SQLite schema.")
        except Exception as exc:  # pragma: no cover
            logger.warning("init_db failed: %s", exc)

    # Warm-load the model so the first request isn't slow (and we fail loudly).
    try:
        service.load()
        logger.info("Model loaded: %s", service.info()["model_name"])
    except Exception as exc:
        logger.warning("Model not loaded at startup: %s", exc)

    if settings.environment == "production" and not settings.api_key:
        logger.warning(
            "SECURITY: running in production with NO API key set - /predict and "
            "/batch are publicly accessible. Set API_KEY to require auth."
        )
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Predicts heart-disease risk from clinical indicators using a "
        "Decision Tree / KNN pipeline (scikit-learn). Trained on the CDC "
        "personal-key-indicators dataset."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prefix = settings.api_v1_prefix
app.include_router(health.router, prefix=prefix)
app.include_router(predict.router, prefix=prefix)
app.include_router(batch.router, prefix=prefix)


# Optionally serve the built React frontend from the same origin. In a
# single-container deployment (see Dockerfile.web) FRONTEND_DIST points at the
# Vite build output, so the SPA is served at "/" while the API stays under
# /api/v1. When it is absent (local docker-compose, tests, API-only deploys) we
# expose a small JSON index at "/" instead. The mount is added last so it only
# catches paths not already handled by the API routers, /docs or /openapi.json.
_frontend_dist = Path(
    os.getenv(
        "FRONTEND_DIST",
        str(Path(__file__).resolve().parent.parent / "frontend_dist"),
    )
)

if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
else:

    @app.get("/", tags=["root"])
    def root() -> dict:
        return {
            "service": settings.app_name,
            "docs": "/docs",
            "health": f"{prefix}/health",
            "predict": f"{prefix}/predict",
        }
