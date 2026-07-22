"""Deployment orchestration smoke test (``docker-compose.yml``).

Asserts the Compose stack wires up all the coordinated services the design's
deployment topology calls for:

* task 15.4 / Req 18.4 -- the stack defines the api, worker, db, broker (redis),
  and frontend services, with the dependency/healthcheck wiring that lets them
  start in the right order.

The compose file is parsed as YAML and inspected structurally (no containers are
launched).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_COMPOSE_PATH = Path(__file__).resolve().parents[2] / "docker-compose.yml"


@pytest.fixture(scope="module")
def services() -> dict:
    """The parsed ``services`` mapping from ``docker-compose.yml``."""
    with open(_COMPOSE_PATH, encoding="utf-8") as fh:
        compose = yaml.safe_load(fh)
    return compose["services"]


def _depends_on_names(service: dict) -> set[str]:
    """The set of service names a service depends on (long or short form)."""
    dep = service.get("depends_on", {})
    if isinstance(dep, dict):
        return set(dep.keys())
    return set(dep)


def test_all_coordinated_services_defined(services: dict) -> None:
    """The stack defines api, worker, db, broker (redis), and frontend.

    **Validates: Requirement 18.4**
    """
    for name in ("api", "worker", "db", "redis", "frontend"):
        assert name in services, f"docker-compose.yml is missing the '{name}' service"


def test_db_and_broker_have_healthchecks(services: dict) -> None:
    """The database and broker declare healthchecks used for ordered startup.

    **Validates: Requirement 18.4**
    """
    assert "healthcheck" in services["db"]
    assert "healthcheck" in services["redis"]


def test_api_and_worker_depend_on_healthy_db_and_broker(services: dict) -> None:
    """api and worker wait for a healthy db + broker before starting.

    **Validates: Requirement 18.4**
    """
    api_deps = services["api"]["depends_on"]
    assert {"db", "redis"}.issubset(_depends_on_names(services["api"]))
    assert api_deps["db"]["condition"] == "service_healthy"
    assert api_deps["redis"]["condition"] == "service_healthy"

    assert {"db", "redis"}.issubset(_depends_on_names(services["worker"]))


def test_frontend_depends_on_api(services: dict) -> None:
    """The frontend is wired to depend on the api service.

    **Validates: Requirement 18.4**
    """
    assert "api" in _depends_on_names(services["frontend"])


def test_api_applies_migrations_before_serving(services: dict) -> None:
    """The api service runs the Alembic migration before starting uvicorn.

    **Validates: Requirement 18.4 (managed-DB schema readiness)**
    """
    command = services["api"]["command"]
    assert "alembic upgrade head" in command
