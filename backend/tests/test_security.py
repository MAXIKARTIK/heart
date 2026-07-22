"""Property + unit tests for the Auth_Guard (``backend/app/core/security.py``).

* Property 19 -- the API-key authorization rule: a protected request is
  permitted exactly when no key is configured or the supplied ``X-API-Key``
  equals the configured key, and rejected with HTTP 401 otherwise (task 11.1).
* Unit test -- the startup security warning emitted when the service runs in a
  production environment with no API key configured (task 11.2).

The property test is a single Hypothesis test running >= 100 examples.
"""
from __future__ import annotations

import asyncio
import logging

import pytest
from fastapi import HTTPException
from hypothesis import given
from hypothesis import strategies as st

from app.core import security
from app.core.config import settings


def _call_guard(x_api_key):
    """Invoke the async ``require_api_key`` dependency synchronously."""
    return asyncio.run(security.require_api_key(x_api_key=x_api_key))


# --------------------------------------------------------------------------- #
# Property 19: API-key authorization rule (task 11.1).
# --------------------------------------------------------------------------- #
# Feature: heart-disease-prediction, Property 19: request permitted iff no key configured or supplied X-API-Key equals configured key, else 401
@given(
    configured=st.one_of(st.just(""), st.text(min_size=1, max_size=32)),
    supplied=st.one_of(st.none(), st.text(min_size=0, max_size=32)),
)
def test_api_key_authorization_rule(configured: str, supplied) -> None:
    """The guard permits iff no key is configured or the header matches, else 401.

    **Validates: Property 19; Requirements 16.1, 16.2, 16.3**
    """
    original = settings.api_key
    settings.api_key = configured
    try:
        permitted_expected = (not configured) or (supplied == configured)

        if permitted_expected:
            # No exception: the request is allowed to proceed.
            assert _call_guard(supplied) is None
        else:
            with pytest.raises(HTTPException) as excinfo:
                _call_guard(supplied)
            assert excinfo.value.status_code == 401
            assert excinfo.value.detail  # descriptive message present
    finally:
        settings.api_key = original


# --------------------------------------------------------------------------- #
# Unit test: production-without-key warning (task 11.2).
# --------------------------------------------------------------------------- #
def test_production_without_api_key_emits_security_warning(
    test_db, set_settings, caplog
) -> None:
    """Startup logs a security warning in production when no API key is set.

    Drives the application ``lifespan`` (via ``TestClient``) with
    ``environment == "production"`` and an empty ``api_key`` and asserts the
    warning naming the publicly-accessible endpoints is emitted.

    **Validates: Requirement 16.4**
    """
    from fastapi.testclient import TestClient

    from app.db import get_db
    from app.main import app

    set_settings(environment="production", api_key="")

    def _override_get_db():
        db = test_db()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    try:
        with caplog.at_level(logging.WARNING, logger="heart.api"):
            with TestClient(app):  # entering runs the startup lifespan
                pass
    finally:
        app.dependency_overrides.clear()

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "production" in msg and "API key" in msg for msg in warnings
    ), f"expected a production/no-key security warning, got: {warnings}"
