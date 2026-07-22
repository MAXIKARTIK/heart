"""Unit tests for application Settings (``backend/app/core/config.py``).

Covers the configuration criteria the design routes to unit tests:

* task 15.1 / Req 18.1 -- ``Settings`` reads the database URL, model path,
  broker connection, API key, and CORS origins from the environment.
* task 15.1 / Req 18.2 -- ``celery_enabled`` is False (so batch is processed
  synchronously) when no broker is configured, and True once one is.
* task 15.1 / Req 18.3 -- ``cors_origin_list`` parses the comma-separated
  origins string (and the "*" wildcard).

Each ``Settings`` instance is built with ``_env_file=None`` so it reads only the
explicitly-set environment / init values and never a stray ``.env`` on disk.
"""
from __future__ import annotations

from app.core.config import Settings


# --------------------------------------------------------------------------- #
# Req 18.1 -- configuration is read from the environment.
# --------------------------------------------------------------------------- #
def test_settings_read_from_environment(monkeypatch) -> None:
    """Every deployment-relevant field is sourced from environment variables.

    **Validates: Requirement 18.1**
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://u:p@db:5432/heart")
    monkeypatch.setenv("MODEL_PATH", "/models/model.joblib")
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://broker:6379/1")
    monkeypatch.setenv("REDIS_URL", "redis://cache:6379/0")
    monkeypatch.setenv("API_KEY", "s3cret-key")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example,https://b.example")

    s = Settings(_env_file=None)

    assert s.database_url == "postgresql+psycopg2://u:p@db:5432/heart"
    assert s.model_path == "/models/model.joblib"
    assert s.celery_broker_url == "redis://broker:6379/1"
    assert s.redis_url == "redis://cache:6379/0"
    assert s.api_key == "s3cret-key"
    assert s.cors_origins == "https://a.example,https://b.example"


# --------------------------------------------------------------------------- #
# Req 18.2 -- broker presence drives celery_enabled (and sync vs async batch).
# --------------------------------------------------------------------------- #
def test_celery_disabled_and_batch_synchronous_without_broker(monkeypatch) -> None:
    """With no broker configured, ``celery_enabled`` is False -> synchronous batch.

    **Validates: Requirement 18.2**
    """
    monkeypatch.delenv("CELERY_BROKER_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    s = Settings(_env_file=None)

    assert s.celery_broker_url == ""
    assert s.redis_url == ""
    assert s.celery_enabled is False
    assert s.effective_broker == ""


def test_celery_enabled_when_broker_configured(monkeypatch) -> None:
    """A configured broker (redis or explicit celery broker) flips the flag on.

    **Validates: Requirement 18.2**
    """
    monkeypatch.delenv("CELERY_BROKER_URL", raising=False)
    monkeypatch.setenv("REDIS_URL", "redis://cache:6379/0")
    assert Settings(_env_file=None).celery_enabled is True

    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://broker:6379/1")
    s = Settings(_env_file=None)
    assert s.celery_enabled is True
    assert s.effective_broker == "redis://broker:6379/1"


# --------------------------------------------------------------------------- #
# Req 18.3 -- CORS origins parsing.
# --------------------------------------------------------------------------- #
def test_cors_origin_list_wildcard() -> None:
    """A "*" origins string parses to the single-element wildcard list.

    **Validates: Requirement 18.3**
    """
    assert Settings(_env_file=None, cors_origins="*").cors_origin_list == ["*"]


def test_cors_origin_list_parses_and_trims() -> None:
    """A comma-separated origins string parses into a trimmed list.

    **Validates: Requirement 18.3**
    """
    s = Settings(_env_file=None, cors_origins="https://a.example,https://b.example")
    assert s.cors_origin_list == ["https://a.example", "https://b.example"]

    # Surrounding whitespace and empty trailing segments are dropped.
    messy = Settings(
        _env_file=None, cors_origins=" https://a.example ,  https://b.example , "
    )
    assert messy.cors_origin_list == ["https://a.example", "https://b.example"]
