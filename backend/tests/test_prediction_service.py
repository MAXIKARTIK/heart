"""Property + unit tests for the Prediction_Service (``backend/app/ml.py``).

This suite validates the model-serving layer's correctness properties from the
design document:

* Property 9  -- the decision rule is ``probability >= threshold`` (task 9.1).
* Property 10 -- a serialized Model_Bundle round-trips without changing scoring
  (task 9.2).
* Property 11 -- scoring is invariant to input key order (task 9.3).
* Property 12 -- prediction output shape and risk label are consistent (task 9.4).
* Property 13 -- batch scoring equals per-record scoring (task 9.5).

plus the deterministic error / concurrency unit tests (task 9.6): the
``ModelNotLoadedError`` message names the artifact path and the train command,
and concurrent ``load()`` calls load the bundle at most once via the
double-checked lock.

Each property test is a single Hypothesis test running >= 100 examples (the
``default`` profile registered in the root ``conftest.py``). The
serialization-round-trip test fits one small pipeline once (the session-scoped
``model_bundle`` fixture), serializes it a single time, and reuses both across
all generated records to keep cost bounded.
"""
from __future__ import annotations

import threading
import time

import joblib
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Several tests below read the module-level Prediction_Service singleton (or the
# one valid record) through a function-scoped fixture. Those fixtures are only
# ever *read* -- scoring never mutates them -- so reusing one instance across a
# test's generated examples is safe; suppress the function-scoped-fixture health
# check for exactly those tests.
_READ_ONLY_FIXTURE = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)

from heart_ml import config as ml_config

import app.ml as ml_module
from app.ml import ModelNotLoadedError, PredictionService

from tests.strategies import clinical_records


# --------------------------------------------------------------------------- #
# Property 9: the decision rule is probability >= threshold (task 9.1).
# --------------------------------------------------------------------------- #
class _StubPipeline:
    """A stand-in pipeline whose ``predict_proba`` returns caller-chosen scores.

    ``PredictionService.predict_many`` calls ``pipeline.predict_proba(X)[:, 1]``,
    so returning a two-column ``[1 - p, p]`` matrix lets a test pin the exact
    positive-class probability the service scores against a chosen threshold --
    isolating the decision rule from any learned model.
    """

    def __init__(self, positive_probas: list[float]) -> None:
        self._p = np.asarray(positive_probas, dtype=float)

    def predict_proba(self, X):  # noqa: N803  (sklearn signature)
        return np.column_stack([1.0 - self._p, self._p])


def _service_with_stub(positive_probas: list[float], threshold: float) -> PredictionService:
    """A PredictionService whose bundle scores with fixed probabilities."""
    svc = PredictionService()
    svc._bundle = {"pipeline": _StubPipeline(positive_probas), "threshold": float(threshold)}
    return svc


# Feature: heart-disease-prediction, Property 9: class is positive exactly when p >= t (including p == t)
@_READ_ONLY_FIXTURE
@given(
    p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    t=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    at_boundary=st.booleans(),
)
def test_decision_rule_is_probability_ge_threshold(
    p: float, t: float, at_boundary: bool, sample_clinical_record: dict
) -> None:
    """The positive class is assigned exactly when ``p >= t`` (boundary included).

    **Validates: Property 9; Requirements 5.4**
    """
    # Deliberately exercise the boundary ``p == t`` on a fraction of examples so
    # the "including p == t" clause is covered rather than probabilistically skipped.
    if at_boundary:
        p = t

    svc = _service_with_stub([p], t)
    result = svc.predict_many([dict(sample_clinical_record)])[0]

    assert result["prediction"] == int(p >= t)
    if p >= t:
        assert result["prediction"] == 1
        assert result["risk_label"] == "At risk"
    else:
        assert result["prediction"] == 0
        assert result["risk_label"] == "Low risk"


# --------------------------------------------------------------------------- #
# Property 10: Model_Bundle serialization round-trip preserves scoring (task 9.2).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def roundtrip_services(model_bundle, tmp_path_factory):
    """Return ``(in_memory_service, reloaded_service)`` sharing one fitted model.

    The session-scoped ``model_bundle`` is fit once; here it is serialized to
    disk a single time and reloaded through a fresh ``PredictionService``. Both
    the in-memory bundle and the reloaded-from-disk bundle are reused across
    every generated record, so the round-trip cost is paid once per session.
    """
    path = tmp_path_factory.mktemp("bundle") / "model.joblib"
    joblib.dump(model_bundle, path)

    in_memory = PredictionService()
    in_memory._bundle = model_bundle

    reloaded = PredictionService(model_path=path)
    reloaded.load()

    return in_memory, reloaded


# Feature: heart-disease-prediction, Property 10: loading a serialized bundle and scoring yields the same probability and prediction as the in-memory pipeline
@given(record=clinical_records())
def test_serialization_round_trip_preserves_scoring(record, roundtrip_services) -> None:
    """A reloaded bundle scores identically to the in-memory pipeline.

    **Validates: Property 10; Requirements 6.2, 6.3**
    """
    in_memory, reloaded = roundtrip_services

    before = in_memory.predict_many([dict(record)])[0]
    after = reloaded.predict_many([dict(record)])[0]

    assert after["probability"] == pytest.approx(before["probability"], abs=1e-12)
    assert after["prediction"] == before["prediction"]
    assert after["risk_label"] == before["risk_label"]


# --------------------------------------------------------------------------- #
# Property 11: scoring is invariant to input key order (task 9.3).
# --------------------------------------------------------------------------- #
@st.composite
def _record_and_permutation(draw: st.DrawFn) -> tuple[dict, dict]:
    """A record and a re-keyed copy with the same values in a shuffled order."""
    record = dict(draw(clinical_records()))
    keys = list(record.keys())
    permuted_keys = draw(st.permutations(keys))
    permuted = {k: record[k] for k in permuted_keys}
    return record, permuted


# Feature: heart-disease-prediction, Property 11: reordering a record's fields does not change probability or prediction
@_READ_ONLY_FIXTURE
@given(case=_record_and_permutation())
def test_scoring_is_invariant_to_input_key_order(case, loaded_service) -> None:
    """Reordering a record's fields leaves probability and prediction unchanged.

    **Validates: Property 11; Requirements 8.4**
    """
    record, permuted = case

    base = loaded_service.predict_many([record])[0]
    reordered = loaded_service.predict_many([permuted])[0]

    assert reordered["probability"] == pytest.approx(base["probability"], abs=1e-12)
    assert reordered["prediction"] == base["prediction"]
    assert reordered["risk_label"] == base["risk_label"]


# --------------------------------------------------------------------------- #
# Property 12: prediction output shape and risk label are consistent (task 9.4).
# --------------------------------------------------------------------------- #
# Feature: heart-disease-prediction, Property 12: prediction in {0,1}, probability in [0,1], label "At risk" iff prediction==1 else "Low risk"
@_READ_ONLY_FIXTURE
@given(record=clinical_records())
def test_prediction_output_shape_and_risk_label_consistent(record, loaded_service) -> None:
    """Every scored result has a valid prediction, probability, and paired label.

    **Validates: Property 12; Requirements 8.5**
    """
    result = loaded_service.predict_many([dict(record)])[0]

    assert isinstance(result["prediction"], int)
    assert result["prediction"] in (0, 1)

    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0

    if result["prediction"] == 1:
        assert result["risk_label"] == "At risk"
    else:
        assert result["risk_label"] == "Low risk"


# --------------------------------------------------------------------------- #
# Property 13: batch scoring equals per-record scoring (task 9.5).
# --------------------------------------------------------------------------- #
# Feature: heart-disease-prediction, Property 13: predict_many over a list matches scoring each record alone, position by position
@_READ_ONLY_FIXTURE
@given(records=st.lists(clinical_records(), min_size=1, max_size=8))
def test_batch_scoring_equals_per_record_scoring(records, loaded_service) -> None:
    """Each position in a batch result matches scoring that record alone.

    **Validates: Property 13; Requirements 12.3, 8.5**
    """
    batch = loaded_service.predict_many([dict(r) for r in records])
    assert len(batch) == len(records)

    for record, batched in zip(records, batch):
        alone = loaded_service.predict_many([dict(record)])[0]
        assert batched["probability"] == pytest.approx(alone["probability"], abs=1e-12)
        assert batched["prediction"] == alone["prediction"]
        assert batched["risk_label"] == alone["risk_label"]


# --------------------------------------------------------------------------- #
# Unit tests: load error and concurrency (task 9.6).
# --------------------------------------------------------------------------- #
def test_missing_artifact_raises_model_not_loaded_error(tmp_path) -> None:
    """Scoring with no artifact raises ``ModelNotLoadedError`` naming the path
    and the train command.

    **Validates: Requirement 8.2**
    """
    missing = tmp_path / "no_such_model.joblib"
    svc = PredictionService(model_path=missing)

    with pytest.raises(ModelNotLoadedError) as excinfo:
        svc.load()
    message = str(excinfo.value)
    assert str(missing) in message
    assert "python -m heart_ml.train" in message

    # The same error surfaces through predict_many (the scoring entry point).
    with pytest.raises(ModelNotLoadedError):
        svc.predict_many([{name: None for name in ml_config.RAW_FEATURES}])


def test_concurrent_load_loads_bundle_at_most_once(model_bundle, tmp_path, monkeypatch) -> None:
    """Concurrent first-time ``load()`` calls read the artifact at most once.

    The double-checked lock in ``PredictionService.load`` must ensure that when
    many threads race to load a cold service, the bundle is deserialized exactly
    once. We count deserializations by wrapping ``joblib.load`` (with a small
    delay to widen the race window) and assert it fires a single time.

    **Validates: Requirement 8.3**
    """
    path = tmp_path / "model.joblib"
    joblib.dump(model_bundle, path)

    load_calls = 0
    real_load = joblib.load

    def counting_load(p, *args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        time.sleep(0.05)  # widen the window so racing threads overlap
        return real_load(p, *args, **kwargs)

    monkeypatch.setattr(ml_module.joblib, "load", counting_load)

    svc = PredictionService(model_path=path)

    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()  # release all threads simultaneously
        svc.load()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert load_calls == 1
    assert svc.is_loaded
