"""Shared pytest fixtures for the FastAPI backend (``app``) test suite.

This is the *harness* for the backend tests (Prediction_Service, Validation_Layer,
Auth_Guard, the API endpoints, Audit_Log persistence, config/startup smoke tests).
It provides everything those later suites need and nothing they don't:

* :func:`model_bundle`        -- a fully-formed Model_Bundle (every ``design.md``
  key) fit once per session on the tiny dataframe, reused across tests.
* :func:`test_db`             -- an isolated temp-file SQLite database that never
  touches the real ``heart.db``; it repoints ``settings``, the SQLAlchemy engine
  and both ``SessionLocal`` references (``app.db`` and ``app.tasks``) at the temp DB.
* :func:`loaded_service` / :func:`unloaded_service` -- inject the test bundle into
  the module-level :data:`app.ml.service` singleton, or force the "no model
  loaded" state so error paths return HTTP 503.
* :func:`client` / :func:`unloaded_client` -- a FastAPI ``TestClient`` with the
  ``get_db`` dependency overridden onto the temp DB, in the model-loaded and
  model-absent states respectively.
* :func:`set_settings`        -- a helper to monkeypatch ``Settings`` fields
  (``api_key``, ``environment``, ``celery_broker_url`` / ``redis_url``, ...).
* :func:`sample_clinical_record` -- one valid raw record for endpoint tests.

Import-path note
----------------
The backend package is imported as the *top-level* ``app`` package (``from app
import ...``), so the ``backend/`` directory must be on ``sys.path``. The project
root is added too so the shared ``tests`` package (``tests.strategies``,
``tests.conftest.build_tiny_dataframe``) and the ``heart_ml`` library resolve even
when this suite is run on its own (``python -m pytest backend/tests/``).
``backend/tests`` is deliberately *not* a package: adding ``backend/tests/__init__.py``
would make it a second top-level ``tests`` package and collide with the
project-root ``tests`` package.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Import path setup -- must run before any ``app.*`` import below.
# --------------------------------------------------------------------------- #
_BACKEND_DIR = Path(__file__).resolve().parents[1]   # .../backend
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # project root
for _path in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pytest


# --------------------------------------------------------------------------- #
# Model_Bundle
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def model_bundle() -> dict:
    """A complete Model_Bundle fit on the tiny dataframe (session-scoped).

    Mirrors how ``heart_ml.train`` assembles the deployable bundle: it fits a
    small Decision Tree pipeline, picks an F1-optimal threshold, computes the
    six evaluation metrics, and packs the fitted pipeline together with all the
    feature/metadata keys the serving layer (``PredictionService.info``) and the
    ``ModelInfo`` schema read. Fitting happens once per test session and the
    resulting dict is reused (tests only read it).

    Contains every key documented in ``design.md``: ``pipeline``, ``threshold``,
    ``model_name``, ``raw_features``, ``numeric_features``,
    ``categorical_features``, ``category_values``, ``metrics``, ``trained_at``,
    ``sklearn_version``, ``n_train`` and ``positive_rate``.
    """
    from datetime import datetime, timezone

    import sklearn

    from heart_ml import config as ml_config
    from heart_ml.data import split_features_target
    from heart_ml.pipeline import decision_tree_pipeline
    from heart_ml.train import best_f1_threshold, compute_metrics
    from tests.conftest import build_tiny_dataframe

    df = build_tiny_dataframe()
    X, y = split_features_target(df)

    # A shallow tree keeps fitting instant and the serialized artifact tiny; the
    # goal here is a *valid* bundle for serving tests, not a strong model.
    pipeline = decision_tree_pipeline(max_depth=5, min_samples_leaf=5)
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X)[:, 1]
    threshold = best_f1_threshold(y, proba)
    y_pred = (proba >= threshold).astype(int)
    metrics = compute_metrics(y, y_pred, proba)

    return {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "model_name": "DecisionTree",
        "raw_features": list(ml_config.RAW_FEATURES),
        "numeric_features": list(ml_config.NUMERIC_FEATURES),
        "categorical_features": list(ml_config.CATEGORICAL_FEATURES),
        "category_values": {k: list(v) for k, v in ml_config.CATEGORY_VALUES.items()},
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(X)),
        "positive_rate": float(y.mean()),
    }


# --------------------------------------------------------------------------- #
# Isolated database
# --------------------------------------------------------------------------- #
@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """An isolated, temp-file SQLite database with all tables created.

    Yields the session factory (a ``sessionmaker``) bound to the temp DB. The
    fixture repoints everything the app uses so **no test ever touches the real
    ``heart.db``**:

    * ``settings.database_url`` -> the temp ``sqlite:///`` URL,
    * ``app.db.engine`` / ``app.db.SessionLocal`` -> the temp engine/factory
      (so ``get_db`` and ``init_db`` use the temp DB), and
    * ``app.tasks.SessionLocal`` -> the temp factory (the Celery batch worker
      imported it by name, so it must be patched separately).

    Function-scoped: every test gets a pristine database.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app import db as app_db
    from app import models as app_models  # noqa: F401  (register tables on Base)
    from app import tasks as app_tasks
    from app.core.config import settings
    from app.db import Base

    db_path = tmp_path / "test_heart.db"
    url = f"sqlite:///{db_path}"
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
        future=True,
    )
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(settings, "database_url", url, raising=False)
    monkeypatch.setattr(app_db, "engine", engine, raising=False)
    monkeypatch.setattr(app_db, "SessionLocal", TestSessionLocal, raising=False)
    monkeypatch.setattr(app_tasks, "SessionLocal", TestSessionLocal, raising=False)

    try:
        yield TestSessionLocal
    finally:
        engine.dispose()


# --------------------------------------------------------------------------- #
# Prediction_Service singleton state
# --------------------------------------------------------------------------- #
@pytest.fixture
def loaded_service(model_bundle, monkeypatch):
    """Inject the test :func:`model_bundle` into the ``app.ml.service`` singleton.

    ``monkeypatch`` restores the singleton's previous ``_bundle`` after the test,
    so state never leaks between tests.
    """
    from app.ml import service

    monkeypatch.setattr(service, "_bundle", model_bundle, raising=False)
    return service


@pytest.fixture
def unloaded_service(tmp_path, monkeypatch):
    """Force the ``app.ml.service`` singleton into the "no model loaded" state.

    Clears any cached bundle and points the service at a non-existent artifact,
    so ``service.load()`` / ``predict_many`` raise ``ModelNotLoadedError`` -- the
    condition the API maps to HTTP 503.
    """
    from app.ml import service

    missing = tmp_path / "no_such_model.joblib"
    monkeypatch.setattr(service, "_bundle", None, raising=False)
    monkeypatch.setattr(service, "_model_path", missing, raising=False)
    return service


# --------------------------------------------------------------------------- #
# FastAPI TestClient
# --------------------------------------------------------------------------- #
def _install_db_override(app, session_factory) -> None:
    """Override the app's ``get_db`` dependency to use the temp DB session."""
    from app.db import get_db

    def _override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture
def client(test_db, loaded_service):
    """A ``TestClient`` with the model loaded and the temp DB wired in.

    Entering the client runs the app lifespan (SQLite schema init + model
    warm-load); ``get_db`` is overridden onto :func:`test_db` so every request
    reads/writes the isolated database.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    _install_db_override(app, test_db)
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def unloaded_client(test_db, unloaded_service):
    """A ``TestClient`` in the "no model loaded" state (for HTTP 503 paths)."""
    from fastapi.testclient import TestClient

    from app.main import app

    _install_db_override(app, test_db)
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Settings overrides
# --------------------------------------------------------------------------- #
@pytest.fixture
def set_settings(monkeypatch):
    """Return a helper that monkeypatches fields on the shared ``Settings``.

    Every module reads the same cached ``settings`` instance, so patching its
    attributes is visible everywhere. ``monkeypatch`` restores them afterwards.
    Example::

        def test_requires_key(set_settings, client):
            set_settings(api_key="secret")
            assert client.post("/api/v1/predict", json=...).status_code == 401

    Handy keys: ``api_key``, ``environment``, ``celery_broker_url``,
    ``redis_url``, ``cors_origins``.
    """
    from app.core.config import settings

    def _apply(**overrides):
        for key, value in overrides.items():
            monkeypatch.setattr(settings, key, value, raising=False)
        return settings

    return _apply


# --------------------------------------------------------------------------- #
# Convenience data
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_clinical_record() -> dict:
    """One valid raw Clinical_Features record (the documented schema example)."""
    from app.schemas import ClinicalFeatures

    example = ClinicalFeatures.model_config["json_schema_extra"]["example"]
    return dict(example)
