"""Property-based tests for the pure helper functions in ``heart_ml.train``.

This module validates the training pipeline's small, side-effect-free helpers
against the correctness properties in the design document. It currently covers
Property 5 (the stratified-subsample cap); the remaining train-function
properties (6, 7, 8) are added alongside it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from heart_ml import config
from heart_ml.train import stratified_subsample


@st.composite
def subsample_cases(draw: st.DrawFn) -> tuple[pd.DataFrame, pd.Series, int]:
    """Generate ``(X, y, cap)`` triples spanning the three cap regimes.

    The dataset always carries both classes with at least two members each, so
    the underlying two-class stratified split is well-defined. The cap ``c`` is
    drawn to land in one of three regimes with roughly equal frequency:

    * ``c <= 0``      -- pass-through (the full dataset is returned).
    * ``0 < c < n``   -- an active cap; constrained to ``2 <= c <= n - 2`` so a
      two-class stratified split can allocate both partitions.
    * ``c >= n``      -- pass-through (the full dataset is returned).
    """
    n = draw(st.integers(min_value=4, max_value=60))
    # Both classes present with >= 2 members each (=> stratifiable).
    n_pos = draw(st.integers(min_value=2, max_value=n - 2))
    regime = draw(st.sampled_from(["nonpositive", "active", "at_or_above"]))
    if regime == "nonpositive":
        c = draw(st.integers(min_value=-5, max_value=0))
    elif regime == "active":
        c = draw(st.integers(min_value=2, max_value=n - 2))
    else:  # at_or_above
        c = draw(st.integers(min_value=n, max_value=n + 10))

    X = pd.DataFrame({"feature": np.arange(n)})
    y = pd.Series([1] * n_pos + [0] * (n - n_pos))
    return X, y, c


# Feature: heart-disease-prediction, Property 5: stratified_subsample returns min(c,n) rows when 0<c<n, else the full dataset
@given(case=subsample_cases())
def test_stratified_subsample_respects_cap(
    case: tuple[pd.DataFrame, pd.Series, int]
) -> None:
    """The returned size is ``min(c, n)`` for ``0 < c < n``, else the full ``n``.

    **Validates: Property 5; Requirements 4.4**
    """
    X, y, c = case
    n = len(X)

    X_s, y_s = stratified_subsample(X, y, c, config.RANDOM_STATE)

    expected = min(c, n) if 0 < c < n else n
    assert len(X_s) == expected
    assert len(y_s) == expected


# --------------------------------------------------------------------------- #
# Property 6: best model selected by maximum F1 (Task 5.2).
# --------------------------------------------------------------------------- #
from heart_ml.train import best_f1_threshold, compute_metrics


def _select_best_model(results: dict) -> str:
    """The Training_Pipeline's model-selection rule (from ``train.main``):
    pick the candidate whose test F1 is maximal."""
    return max(results, key=lambda n: results[n]["metrics"]["f1"])


@st.composite
def candidate_result_maps(draw: st.DrawFn) -> dict:
    """A map of >= 2 candidate models to result dicts with distinct F1 scores.

    F1 values are kept distinct so the maximum is unambiguous, mirroring the
    ``results`` structure that ``train.main`` selects over.
    """
    n = draw(st.integers(min_value=2, max_value=5))
    names = [f"model_{i}" for i in range(n)]
    f1s = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=n, max_size=n, unique=True,
        )
    )
    return {name: {"metrics": {"f1": f1}} for name, f1 in zip(names, f1s)}


# Feature: heart-disease-prediction, Property 6: the deployed model is the candidate with maximal F1
@given(results=candidate_result_maps())
def test_best_model_selected_by_maximum_f1(results: dict) -> None:
    """The selected model is the candidate with the maximal F1 score.

    **Validates: Property 6; Requirements 4.5**
    """
    selected = _select_best_model(results)

    best_f1 = max(r["metrics"]["f1"] for r in results.values())
    assert results[selected]["metrics"]["f1"] == best_f1
    # No other candidate has a strictly higher F1.
    for name, r in results.items():
        assert results[selected]["metrics"]["f1"] >= r["metrics"]["f1"]


# --------------------------------------------------------------------------- #
# Property 7: computed metrics are well-formed (Task 5.3).
# --------------------------------------------------------------------------- #
@st.composite
def label_proba_vectors(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ``(y_true, y_pred, proba)`` with both classes present.

    ``roc_auc_score`` and ``average_precision_score`` require at least one of
    each class in ``y_true``; the strategy therefore guarantees both a 0 and a
    1 are present. ``proba`` is a probability in [0, 1] and ``y_pred`` is a
    thresholded 0/1 label vector.
    """
    n = draw(st.integers(min_value=2, max_value=40))
    y_true = draw(
        st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n)
    )
    # Ensure both classes appear.
    if 0 not in y_true:
        y_true[draw(st.integers(min_value=0, max_value=n - 1))] = 0
    if 1 not in y_true:
        y_true[draw(st.integers(min_value=0, max_value=n - 1))] = 1
    proba = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n, max_size=n,
        )
    )
    proba_arr = np.array(proba)
    y_pred = (proba_arr >= 0.5).astype(int)
    return np.array(y_true), y_pred, proba_arr


# Feature: heart-disease-prediction, Property 7: compute_metrics returns all six keys, each in [0,1]
@given(vectors=label_proba_vectors())
def test_compute_metrics_are_well_formed(
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """``compute_metrics`` returns all six metric keys, each within [0, 1].

    **Validates: Property 7; Requirements 4.6**
    """
    y_true, y_pred, proba = vectors

    metrics = compute_metrics(y_true, y_pred, proba)

    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert set(metrics.keys()) == expected_keys
    for key, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{key}={value} is outside [0, 1]"


# --------------------------------------------------------------------------- #
# Property 8: the chosen threshold maximizes F1 (Task 5.4).
# --------------------------------------------------------------------------- #
from sklearn.metrics import f1_score, precision_recall_curve


@st.composite
def validation_label_proba(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    """Generate a validation ``(y_true, proba)`` pair with both classes present
    and at least two distinct probability values (so PR-curve candidates exist)."""
    n = draw(st.integers(min_value=3, max_value=40))
    y_true = draw(st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n))
    if 0 not in y_true:
        y_true[0] = 0
    if 1 not in y_true:
        y_true[-1] = 1
    proba = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n, max_size=n,
        )
    )
    # Guarantee at least two distinct scores.
    if len(set(proba)) < 2:
        proba[0] = 0.1
        proba[-1] = 0.9
    return np.array(y_true), np.array(proba)


# Feature: heart-disease-prediction, Property 8: best_f1_threshold F1 >= F1 at every other PR-curve threshold; 0.5 when no candidates
@given(case=validation_label_proba())
def test_best_f1_threshold_maximizes_f1(
    case: tuple[np.ndarray, np.ndarray]
) -> None:
    """The returned threshold's F1 is >= the F1 at every other PR-curve candidate.

    **Validates: Property 8; Requirements 5.2, 5.3**
    """
    y_true, proba = case

    chosen = best_f1_threshold(y_true, proba)
    chosen_f1 = f1_score(y_true, (proba >= chosen).astype(int), zero_division=0)

    # Independently enumerate the PR-curve candidate thresholds and confirm no
    # candidate yields a strictly higher F1 than the chosen threshold.
    _, _, thresholds = precision_recall_curve(y_true, proba)
    assert thresholds.size > 0
    for t in thresholds:
        cand_f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        assert chosen_f1 >= cand_f1 - 1e-9


# --------------------------------------------------------------------------- #
# Unit test: empty-threshold fallback returns 0.5 (Task 5.5).
# --------------------------------------------------------------------------- #
import heart_ml.train as train_module


def test_best_f1_threshold_falls_back_to_half_when_no_candidates(monkeypatch) -> None:
    """With no PR-curve candidate thresholds, ``best_f1_threshold`` returns 0.5.

    The precision-recall curve always yields at least one threshold for valid,
    non-empty input, so the empty-candidate branch is a defensive guard. We
    exercise it by patching ``precision_recall_curve`` to return an empty
    thresholds array and asserting the safe default is returned.

    **Validates: Requirement 5.3**
    """
    def fake_pr_curve(y_true, proba):
        return np.array([1.0]), np.array([1.0]), np.array([])

    monkeypatch.setattr(train_module, "precision_recall_curve", fake_pr_curve)

    assert best_f1_threshold(np.array([0, 1]), np.array([0.2, 0.8])) == 0.5
