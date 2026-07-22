"""Audit_Log persistence tests (``backend/app/models.py`` + routers/worker).

Every served prediction is written to the ``predictions`` table. These
integration tests drive the API through the ``client`` fixture and then read the
rows back from the *same* isolated temp SQLite database (the ``test_db`` session
factory the client is wired onto) to assert what was persisted:

* task 14.1 -- a single prediction persists a row with ``source="single"`` and
  the returned id points at it (Req 9.2).
* task 14.2 -- each batch record persists with ``source="batch"`` and the row
  faithfully records model_name, threshold, probability, prediction, risk_label,
  source, and the raw Clinical_Features JSON (Req 12.4, 15.1).
* task 14.3 -- each persisted row is assigned a unique id and a creation
  timestamp (Req 15.2).

``client`` and ``test_db`` share one function-scoped temp database (``client``
depends on ``test_db``), so a row written by a request is visible to a query
opened here.
"""
from __future__ import annotations

from datetime import datetime

from app.models import Prediction

API = "/api/v1"


# --------------------------------------------------------------------------- #
# Task 14.1 -- single-prediction logging.
# --------------------------------------------------------------------------- #
def test_single_prediction_is_logged_as_single(
    client, test_db, sample_clinical_record
) -> None:
    """A served single prediction persists a ``source="single"`` row by id.

    **Validates: Requirement 9.2**
    """
    resp = client.post(f"{API}/predict", json=sample_clinical_record)
    assert resp.status_code == 200, resp.text
    row_id = resp.json()["id"]
    assert isinstance(row_id, int)

    session = test_db()
    try:
        row = session.get(Prediction, row_id)
        assert row is not None, "returned id does not point at a persisted row"
        assert row.id == row_id
        assert row.source == "single"
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Task 14.2 -- batch logging + raw-input fidelity.
# --------------------------------------------------------------------------- #
def test_batch_records_logged_with_full_fidelity(
    client, test_db, sample_clinical_record
) -> None:
    """Each batch record persists as ``source="batch"`` with faithful fields.

    Asserts the stored row records every documented field and the *raw* clinical
    inputs verbatim in the JSON ``features`` column.

    **Validates: Requirements 12.4, 15.1**
    """
    records = [dict(sample_clinical_record) for _ in range(3)]
    resp = client.post(f"{API}/batch", json={"records": records})
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]

    session = test_db()
    try:
        rows = session.query(Prediction).order_by(Prediction.id).all()
        assert len(rows) == 3
        for row, res in zip(rows, results):
            # 12.4 -- request source is recorded as "batch".
            assert row.source == "batch"
            # 15.1 -- the persisted row records all served fields...
            assert row.model_name
            assert isinstance(row.threshold, float)
            assert row.probability == res["probability"]
            assert row.prediction == res["prediction"]
            assert row.risk_label == res["risk_label"]
            # ...and the raw Clinical_Features inputs verbatim.
            assert row.features == sample_clinical_record
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Task 14.3 -- unique id + creation timestamp.
# --------------------------------------------------------------------------- #
def test_each_prediction_gets_unique_id_and_timestamp(
    client, test_db, sample_clinical_record
) -> None:
    """Two served predictions get distinct ids and populated creation timestamps.

    **Validates: Requirement 15.2**
    """
    id1 = client.post(f"{API}/predict", json=sample_clinical_record).json()["id"]
    id2 = client.post(f"{API}/predict", json=sample_clinical_record).json()["id"]
    assert id1 != id2, "ids must be unique per persisted prediction"

    session = test_db()
    try:
        row1 = session.get(Prediction, id1)
        row2 = session.get(Prediction, id2)
        for row in (row1, row2):
            assert row is not None
            assert row.created_at is not None
            assert isinstance(row.created_at, datetime)
    finally:
        session.close()
