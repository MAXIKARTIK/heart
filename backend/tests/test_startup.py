"""Startup / lifespan smoke tests (``backend/app/main.py``).

These exercise the one-time application-startup behavior the design routes to
smoke tests:

* task 15.2 / Req 8.1  -- the model is warm-loaded during the app lifespan, so
  the first prediction request incurs no load delay.
* task 15.2 / Req 15.3 -- for a SQLite deployment the ``predictions`` table is
  auto-created at startup.

Both drive the real ``app`` through a ``TestClient`` (entering the client runs
the ``lifespan``) but wire it onto throwaway state so nothing touches the real
``heart.db`` or the shared model singleton beyond the test.
"""
from __future__ import annotations

import joblib
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker


# --------------------------------------------------------------------------- #
# Req 8.1 -- the model warm-loads during the lifespan.
# --------------------------------------------------------------------------- #
def test_model_warm_loads_during_lifespan(
    model_bundle, tmp_path, test_db, monkeypatch
) -> None:
    """Entering the app loads the model before any request is served.

    A cold ``PredictionService`` (``_bundle is None``) is pointed at an on-disk
    bundle. After the lifespan runs -- and before any request is made -- the
    service reports loaded, proving the warm-load happened at startup.

    **Validates: Requirement 8.1**
    """
    from app.db import get_db
    from app.main import app
    from app.ml import service

    path = tmp_path / "model.joblib"
    joblib.dump(model_bundle, path)

    # Force the singleton cold and point it at the artifact on disk.
    monkeypatch.setattr(service, "_bundle", None, raising=False)
    monkeypatch.setattr(service, "_model_path", path, raising=False)
    assert service.is_loaded is False  # cold before startup

    def _override_get_db():
        db = test_db()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    try:
        with TestClient(app):
            # Lifespan has run; no request issued yet.
            assert service.is_loaded is True
    finally:
        app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Req 15.3 -- SQLite predictions table is auto-created at startup.
# --------------------------------------------------------------------------- #
def test_sqlite_predictions_table_auto_created_at_startup(tmp_path, monkeypatch) -> None:
    """A fresh SQLite database gains the ``predictions`` table during startup.

    The app is pointed at a brand-new SQLite file with **no** tables created up
    front; entering the ``TestClient`` runs the lifespan (which calls
    ``init_db`` for SQLite) and the table then exists.

    **Validates: Requirement 15.3**
    """
    from app import db as app_db
    from app import tasks as app_tasks
    from app.core.config import settings
    from app.main import app

    db_path = tmp_path / "startup.db"
    url = f"sqlite:///{db_path}"
    engine = create_engine(
        url, connect_args={"check_same_thread": False}, future=True
    )
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    # Point the app at the fresh DB WITHOUT pre-creating any tables.
    monkeypatch.setattr(settings, "database_url", url, raising=False)
    monkeypatch.setattr(app_db, "engine", engine, raising=False)
    monkeypatch.setattr(app_db, "SessionLocal", TestSessionLocal, raising=False)
    monkeypatch.setattr(app_tasks, "SessionLocal", TestSessionLocal, raising=False)

    try:
        assert "predictions" not in inspect(engine).get_table_names()

        with TestClient(app):
            pass  # entering runs the startup lifespan -> init_db()

        assert "predictions" in inspect(engine).get_table_names()
    finally:
        engine.dispose()
