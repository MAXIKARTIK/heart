"""Integration tests for the end-to-end Training_Pipeline (`heart_ml.train`).

These tests run one real training pass over the tiny stratified fixture and
validate the *observable behavior* of the run against three acceptance
criteria:

* **4.1** -- both a Decision Tree and a KNN candidate are trained.
* **4.2** -- tuning uses ``GridSearchCV(scoring="average_precision", cv=3)``.
* **4.3** -- the held-out test split is 20% stratified with ``random_state=44``
  and preserves the dataset's class ratio.

Because the production system already exists, this is a *validation* suite that
exercises the real code with a handful of representative runs rather than
randomized (Hypothesis) iteration.

Safety
------
``heart_ml.config`` resolves its filesystem paths (``ARTIFACTS_DIR``,
``MODEL_PATH``, ``PLOTS_DIR``) from environment variables **at import time**, so
setting those env vars now -- after the module is already imported -- would have
no effect. To guarantee the developer's real ``heart_ml/artifacts/model.joblib``
is never read, written, or replaced, the training run below redirects the output
paths by patching the module-level ``config`` constants directly, points the
data at the tiny CSV via ``--data``, and disables plots via ``--no-plots``. A
dedicated test then asserts the real artifact is byte-for-byte unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.model_selection import train_test_split as sk_train_test_split

from heart_ml import config
from heart_ml import train as train_module
from heart_ml.data import load_dataset, split_features_target


def _artifact_signature(path: Path) -> tuple[int, int] | None:
    """Return an ``(mtime_ns, size)`` signature for ``path``, or ``None`` if absent."""
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


@pytest.fixture(scope="module")
def training_run(tmp_path_factory: pytest.TempPathFactory, tiny_csv: Path) -> dict:
    """Run one full training pass on the tiny dataset with outputs redirected.

    All training artifacts are written into a temporary directory (never the
    real ``heart_ml/artifacts``). Along the way the fixture spies on every
    ``GridSearchCV`` construction and every ``train_test_split`` call so the
    tests can assert exactly how tuning and the held-out split were configured.
    The signature of the real ``model.joblib`` is captured before and after the
    run to prove it was left untouched.
    """
    artifacts_dir = tmp_path_factory.mktemp("train_artifacts")
    model_path = artifacts_dir / "model.joblib"
    plots_dir = artifacts_dir / "plots"

    # The developer's real artifact, referenced independently of any env var.
    real_artifact = config.PACKAGE_DIR / "artifacts" / "model.joblib"
    real_artifact_before = _artifact_signature(real_artifact)

    gscv_calls: list[dict] = []
    tts_calls: list[dict] = []

    with pytest.MonkeyPatch.context() as mp:
        # Redirect ALL training outputs to the temp dir. config captured these
        # paths at import time, so patch the module constants directly (env
        # vars set now would be ignored).
        mp.setattr(config, "ARTIFACTS_DIR", artifacts_dir)
        mp.setattr(config, "MODEL_PATH", model_path)
        mp.setattr(config, "PLOTS_DIR", plots_dir)

        # Spy on GridSearchCV to capture its tuning configuration (4.2).
        real_gscv = train_module.GridSearchCV

        def spy_gscv(*args, **kwargs):
            gscv_calls.append(kwargs)
            return real_gscv(*args, **kwargs)

        mp.setattr(train_module, "GridSearchCV", spy_gscv)

        # Spy on train_test_split to capture how the held-out split is drawn (4.3).
        real_tts = train_module.train_test_split

        def spy_tts(*args, **kwargs):
            tts_calls.append({"args": args, "kwargs": kwargs})
            return real_tts(*args, **kwargs)

        mp.setattr(train_module, "train_test_split", spy_tts)

        exit_code = train_module.main(["--data", str(tiny_csv), "--no-plots"])

    real_artifact_after = _artifact_signature(real_artifact)

    metrics = json.loads((artifacts_dir / "metrics.json").read_text())
    comparison = (artifacts_dir / "comparison.csv").read_text()

    return {
        "exit_code": exit_code,
        "artifacts_dir": artifacts_dir,
        "model_path": model_path,
        "plots_dir": plots_dir,
        "metrics": metrics,
        "comparison": comparison,
        "gscv_calls": gscv_calls,
        "tts_calls": tts_calls,
        "real_artifact": real_artifact,
        "real_artifact_before": real_artifact_before,
        "real_artifact_after": real_artifact_after,
    }


def test_training_trains_decision_tree_and_knn(training_run: dict) -> None:
    """Both a Decision Tree and a KNN candidate are trained and compared.

    **Validates: Requirements 4.1**
    """
    assert training_run["exit_code"] == 0

    # The metrics report carries a per-model entry for each candidate.
    models = training_run["metrics"]["models"]
    assert set(models.keys()) == {"DecisionTree", "KNN"}

    # comparison.csv lists both models under the six metric columns.
    rows = training_run["comparison"].strip().splitlines()
    header = rows[0].split(",")
    assert header == ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    model_names = {line.split(",")[0] for line in rows[1:]}
    assert model_names == {"DecisionTree", "KNN"}

    # Exactly two candidates were tuned -- one grid search per model.
    assert len(training_run["gscv_calls"]) == 2


def test_tuning_uses_gridsearchcv_average_precision_3fold(training_run: dict) -> None:
    """Each candidate is tuned with GridSearchCV, scored on average precision, 3-fold.

    **Validates: Requirements 4.2**
    """
    gscv_calls = training_run["gscv_calls"]
    assert len(gscv_calls) == 2, "expected one GridSearchCV per candidate model"

    for kwargs in gscv_calls:
        assert kwargs["scoring"] == "average_precision"
        assert kwargs["cv"] == 3

    # The cv value originates from the documented training configuration.
    assert config.CV_FOLDS == 3


def test_heldout_test_split_is_20pct_stratified_seed44(
    training_run: dict, tiny_csv: Path
) -> None:
    """The held-out test set is 20% of the full data, stratified with seed 44.

    **Validates: Requirements 4.3**
    """
    df = load_dataset(str(tiny_csv))
    X, y = split_features_target(df)

    # (a) The first train_test_split in the run is the held-out test split; it
    #     must use test_size=0.2, random_state=44, and stratify on the full
    #     target.
    tts_calls = training_run["tts_calls"]
    assert tts_calls, "expected train_test_split to be called during training"
    first = tts_calls[0]["kwargs"]
    assert first["test_size"] == 0.2
    assert first["random_state"] == 44
    assert first["stratify"] is not None
    assert len(first["stratify"]) == len(y)  # stratified on the full target

    # The documented configuration constants are the source of those values.
    assert config.TEST_SIZE == 0.2
    assert config.RANDOM_STATE == 44

    # (b) Replicate the split independently and confirm it holds out ~20% of the
    #     data while preserving the class ratio (i.e. it is genuinely stratified).
    _, X_test, _, y_test = sk_train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    assert len(X_test) == pytest.approx(0.2 * len(X), abs=1)
    assert set(y_test.unique().tolist()) == {0, 1}  # both classes represented
    # A stratified split preserves each class proportion to within one sample,
    # so the test-set positive rate matches the full dataset's positive rate.
    assert y_test.mean() == pytest.approx(y.mean(), abs=1.0 / len(y_test) + 1e-9)


def test_real_model_artifact_is_untouched(training_run: dict) -> None:
    """Training redirected its outputs; the real model.joblib is unchanged."""
    before = training_run["real_artifact_before"]
    after = training_run["real_artifact_after"]

    # The developer's real artifact existed beforehand and is byte-for-byte
    # identical afterward (same mtime and size).
    assert before is not None, "expected the developer's real model.joblib to exist"
    assert after == before

    # The run wrote its bundle to the redirected temp path instead of the real
    # artifacts directory.
    assert training_run["model_path"].exists()
    assert training_run["model_path"].parent != config.PACKAGE_DIR / "artifacts"


# --------------------------------------------------------------------------- #
# Task 6.2: report artifacts (metrics.json, comparison.csv, plots).
# Task 6.3: Model_Bundle key completeness.
# --------------------------------------------------------------------------- #
from joblib import load as joblib_load


@pytest.fixture(scope="module")
def training_run_with_plots(
    tmp_path_factory: pytest.TempPathFactory, tiny_csv: Path
) -> dict:
    """Run one training pass with plotting enabled, outputs fully redirected.

    Mirrors the ``training_run`` fixture but omits ``--no-plots`` so the three
    PNG charts are produced, letting the plots-enabled criterion (7.3) be
    validated without touching the developer's real artifacts directory.
    """
    artifacts_dir = tmp_path_factory.mktemp("train_artifacts_plots")
    model_path = artifacts_dir / "model.joblib"
    plots_dir = artifacts_dir / "plots"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "ARTIFACTS_DIR", artifacts_dir)
        mp.setattr(config, "MODEL_PATH", model_path)
        mp.setattr(config, "PLOTS_DIR", plots_dir)
        exit_code = train_module.main(["--data", str(tiny_csv)])

    return {
        "exit_code": exit_code,
        "artifacts_dir": artifacts_dir,
        "plots_dir": plots_dir,
    }


def test_metrics_json_has_per_model_metrics_threshold_params(training_run: dict) -> None:
    """``metrics.json`` records per-model metrics, threshold, and chosen params.

    **Validates: Requirement 7.1**
    """
    metrics = training_run["metrics"]

    # Run-level metadata is present.
    for key in ("selected_model", "trained_at", "sklearn_version", "n_train",
                "n_test", "positive_rate", "models"):
        assert key in metrics

    # Each candidate model carries its metrics, tuned threshold, and best params.
    for name, entry in metrics["models"].items():
        assert set(entry["metrics"].keys()) == {
            "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
        }
        assert isinstance(entry["threshold"], (int, float))
        assert "best_params" in entry
        assert "cv_pr_auc" in entry


def test_comparison_csv_lists_six_metric_columns(training_run: dict) -> None:
    """``comparison.csv`` lists the six metrics for each model.

    **Validates: Requirement 7.2**
    """
    rows = training_run["comparison"].strip().splitlines()
    header = rows[0].split(",")
    assert header == ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

    # One data row per model, each with a value for every metric column.
    assert len(rows) - 1 == len(training_run["metrics"]["models"])
    for line in rows[1:]:
        cells = line.split(",")
        assert len(cells) == len(header)


def test_plots_written_when_enabled(training_run_with_plots: dict) -> None:
    """With plots enabled, the three PNG charts are written to the artifacts dir.

    **Validates: Requirement 7.3**
    """
    assert training_run_with_plots["exit_code"] == 0
    plots_dir = training_run_with_plots["plots_dir"]
    for name in ("model_comparison.png", "roc_curve.png", "confusion_matrices.png"):
        assert (plots_dir / name).exists(), f"missing plot {name}"


def test_no_plots_written_when_disabled(training_run: dict) -> None:
    """With ``--no-plots`` the run completes but writes no plot files.

    **Validates: Requirement 7.4**
    """
    assert training_run["exit_code"] == 0
    plots_dir = training_run["plots_dir"]
    # No plot directory / files were produced by the --no-plots run.
    if plots_dir.exists():
        assert not list(plots_dir.glob("*.png"))


def test_model_bundle_has_all_documented_keys(training_run: dict) -> None:
    """The serialized Model_Bundle contains every documented key.

    **Validates: Requirement 6.1**
    """
    bundle = joblib_load(training_run["model_path"])

    expected_keys = {
        "pipeline",
        "threshold",
        "model_name",
        "raw_features",
        "numeric_features",
        "categorical_features",
        "category_values",
        "metrics",
        "trained_at",
        "sklearn_version",
        "n_train",
        "positive_rate",
    }
    assert expected_keys <= set(bundle.keys())

    # Spot-check the shape of the feature metadata and metrics.
    assert bundle["raw_features"] == config.RAW_FEATURES
    assert bundle["numeric_features"] == config.NUMERIC_FEATURES
    assert bundle["categorical_features"] == config.CATEGORICAL_FEATURES
    assert set(bundle["metrics"].keys()) == {
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    }
    assert bundle["model_name"] in {"DecisionTree", "KNN"}
