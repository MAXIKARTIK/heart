"""Endpoint tests for the Prediction_API (``backend/app/routers/*``).

These are integration tests driven through a FastAPI ``TestClient`` (the
``client`` / ``unloaded_client`` fixtures from ``conftest.py``). They cover the
fixed response shapes and deterministic error paths that the design routes to
integration tests rather than to property-based tests:

* task 13.1 -- the single-prediction endpoint (Req 9.1, 9.3).
* task 13.2 -- the health + model-metadata endpoints (Req 14.1, 14.2, 14.3).
* task 13.3 -- the synchronous batch endpoint with no broker (Req 12.3, 12.5).
* task 13.4 -- asynchronous batch dispatch + status polling (Req 12.2, 13.1,
  13.2, 13.3), exercised with Celery in eager mode and the broker "enabled" via
  monkeypatched settings.

The ``client`` fixture has the model loaded, an isolated temp SQLite DB wired in,
and no broker configured (so batch is synchronous by default). ``unloaded_client``
is the same wiring in the "no model loaded" state, used for the HTTP 503 paths.
"""
from __future__ import annotations

API = "/api/v1"

_RESULT_KEYS = {"prediction", "probability", "risk_label"}


def _assert_scored_item(item: dict) -> None:
    """A per-record scored item has the documented shape and paired label."""
    assert item["prediction"] in (0, 1)
    assert 0.0 <= item["probability"] <= 1.0
    if item["prediction"] == 1:
        assert item["risk_label"] == "At risk"
    else:
        assert item["risk_label"] == "Low risk"


# --------------------------------------------------------------------------- #
# Task 13.1 -- single-prediction endpoint.
# --------------------------------------------------------------------------- #
def test_single_prediction_returns_full_payload(client, sample_clinical_record) -> None:
    """POST /predict returns every documented field plus the logged row id.

    **Validates: Requirement 9.1**
    """
    resp = client.post(f"{API}/predict", json=sample_clinical_record)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Every field the PredictionResponse documents must be present.
    for key in (
        "id",
        "prediction",
        "risk_label",
        "probability",
        "threshold",
        "model_name",
        "trained_at",
    ):
        assert key in body, f"missing '{key}' in response {body}"

    _assert_scored_item(body)
    assert isinstance(body["threshold"], float)
    assert 0.0 <= body["threshold"] <= 1.0
    assert isinstance(body["model_name"], str) and body["model_name"]
    assert isinstance(body["trained_at"], str) and body["trained_at"]
    # A single prediction is logged and its id is returned (Req 9.2 surface).
    assert isinstance(body["id"], int)


def test_single_prediction_503_when_model_absent(
    unloaded_client, sample_clinical_record
) -> None:
    """POST /predict returns HTTP 503 with a descriptive detail when no model.

    **Validates: Requirement 9.3**
    """
    resp = unloaded_client.post(f"{API}/predict", json=sample_clinical_record)
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    # The detail is descriptive: it instructs the operator how to train.
    assert "python -m heart_ml.train" in detail


# --------------------------------------------------------------------------- #
# Task 13.2 -- health + model-metadata endpoints.
# --------------------------------------------------------------------------- #
def test_health_liveness_payload(client) -> None:
    """GET /health reports status, environment, model_loaded, async_enabled.

    **Validates: Requirement 14.1**
    """
    resp = client.get(f"{API}/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "environment" in body
    assert body["model_loaded"] is True          # client loads the model
    assert body["async_enabled"] is False         # no broker configured


def test_model_metadata_when_loaded(client) -> None:
    """GET /health/model returns the full model-metadata payload when loaded.

    **Validates: Requirement 14.2**
    """
    resp = client.get(f"{API}/health/model")
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "model_name",
        "threshold",
        "trained_at",
        "sklearn_version",
        "n_train",
        "positive_rate",
        "metrics",
    ):
        assert key in body, f"missing '{key}' in {body}"
    # The six evaluation metrics ride along in the metadata.
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}.issubset(
        body["metrics"]
    )


def test_model_metadata_503_when_absent(unloaded_client) -> None:
    """GET /health/model returns HTTP 503 with a descriptive detail when no model.

    **Validates: Requirement 14.3**
    """
    resp = unloaded_client.get(f"{API}/health/model")
    assert resp.status_code == 503
    assert resp.json()["detail"]


# --------------------------------------------------------------------------- #
# Task 13.3 -- synchronous batch endpoint (no broker).
# --------------------------------------------------------------------------- #
def test_sync_batch_scores_inline(client, sample_clinical_record) -> None:
    """With no broker, POST /batch scores inline and returns per-record results.

    **Validates: Requirement 12.3**
    """
    records = [dict(sample_clinical_record) for _ in range(3)]
    resp = client.post(f"{API}/batch", json={"records": records})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["status"] == "completed"
    assert body["count"] == 3
    assert body["task_id"] is None                 # inline: no async task
    assert body["results"] is not None
    assert len(body["results"]) == 3
    for item in body["results"]:
        assert _RESULT_KEYS.issubset(item)
        _assert_scored_item(item)


def test_sync_batch_503_when_model_absent_and_no_broker(
    unloaded_client, sample_clinical_record
) -> None:
    """POST /batch returns HTTP 503 when no model is loaded and no broker exists.

    **Validates: Requirement 12.5**
    """
    resp = unloaded_client.post(
        f"{API}/batch", json={"records": [dict(sample_clinical_record)]}
    )
    assert resp.status_code == 503
    assert "python -m heart_ml.train" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# Task 13.4 -- asynchronous batch dispatch + polling (Celery eager).
# --------------------------------------------------------------------------- #
def test_async_batch_enqueue_then_poll_success(
    client, set_settings, sample_clinical_record
) -> None:
    """With a broker configured, POST /batch enqueues and polling yields results.

    Celery is already in eager mode (no real broker at import), so configuring a
    broker URL flips ``settings.celery_enabled`` to True -- the API takes the
    async branch -- while ``.delay()`` still executes inline. The enqueue
    response is ``{task_id, status:"queued", count}`` (12.2); polling returns the
    task id and status (13.1) and the per-record results once complete (13.2).

    **Validates: Requirements 12.2, 13.1, 13.2**
    """
    set_settings(redis_url="memory://")  # celery_enabled -> True

    records = [dict(sample_clinical_record) for _ in range(4)]
    resp = client.post(f"{API}/batch", json={"records": records})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # 12.2 -- enqueue response shape.
    assert body["status"] == "queued"
    assert body["count"] == 4
    assert isinstance(body["task_id"], str) and body["task_id"]

    # 13.1 / 13.2 -- poll the job and receive its id, status, and results.
    poll = client.get(f"{API}/batch/{body['task_id']}")
    assert poll.status_code == 200, poll.text
    pbody = poll.json()
    assert pbody["task_id"] == body["task_id"]
    assert pbody["status"] == "SUCCESS"
    assert pbody["results"] is not None
    assert len(pbody["results"]) == 4
    for item in pbody["results"]:
        _assert_scored_item(item)


def test_batch_status_404_when_async_disabled(client) -> None:
    """Polling a task while no broker is configured returns HTTP 404.

    **Validates: Requirement 13.3**
    """
    resp = client.get(f"{API}/batch/does-not-exist")
    assert resp.status_code == 404
    assert "disabled" in resp.json()["detail"].lower()
