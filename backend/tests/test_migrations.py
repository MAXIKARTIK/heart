"""Alembic migration smoke test (``backend/alembic/versions/0001_initial.py``).

For a managed (PostgreSQL) deployment the schema is applied via Alembic before
requests are served. This test applies the initial migration against a scratch
SQLite database and asserts the resulting schema matches what the ORM /
Audit_Log expects:

* task 15.3 / Req 15.4 -- the ``predictions`` table and its ``created_at`` index
  exist after ``alembic upgrade head``.

The migration is driven in-process via the Alembic API. The Alembic ``Config``
is built **without** the ``alembic.ini`` file so importing it here does not
reconfigure the test session's logging; ``env.py`` reads the target database
from the ``DATABASE_URL`` environment variable, which the test sets to the
scratch DB.
"""
from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

_BACKEND_DIR = Path(__file__).resolve().parents[1]


def test_initial_migration_creates_predictions_table_and_index(
    tmp_path, monkeypatch
) -> None:
    """``alembic upgrade head`` creates the predictions table + created_at index.

    **Validates: Requirement 15.4**
    """
    db_path = tmp_path / "migrated.db"
    url = f"sqlite:///{db_path}"
    # env.py resolves the URL from DATABASE_URL (falling back to app settings).
    monkeypatch.setenv("DATABASE_URL", url)

    # Build the Alembic config programmatically (no .ini -> no logging reconfig);
    # point it at the backend's migration scripts.
    cfg = Config()
    cfg.set_main_option("script_location", str(_BACKEND_DIR / "alembic"))

    command.upgrade(cfg, "head")

    engine = create_engine(url, future=True)
    try:
        inspector = inspect(engine)
        assert "predictions" in inspector.get_table_names()

        index_names = {ix["name"] for ix in inspector.get_indexes("predictions")}
        assert "ix_predictions_created_at" in index_names
    finally:
        engine.dispose()
