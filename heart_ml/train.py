"""Train, tune and compare Decision Tree + KNN, then serialize the best model.

Run with::

    python -m heart_ml.train            # or: heart-train

Outputs (into ``heart_ml/artifacts/``):
    model.joblib      -- deployable bundle: fitted pipeline + decision threshold
    metrics.json      -- full metrics for both models + metadata
    comparison.csv    -- tabular metric comparison
    plots/*.png       -- metric bar chart, ROC curves, confusion matrices

Design notes
------------
* The target is imbalanced (~8.5% positive). We therefore (a) use
  ``class_weight='balanced'`` where supported, (b) tune with average-precision
  (PR-AUC) rather than accuracy, and (c) pick a decision threshold that
  maximizes F1 on a validation split instead of naively using 0.5.
* All preprocessing is inside the pipeline and fit only on training folds, so
  there is no leakage into the test set.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from . import config
from .data import load_dataset, split_features_target
from .pipeline import (
    DT_PARAM_GRID,
    KNN_PARAM_GRID,
    decision_tree_pipeline,
    knn_pipeline,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def best_f1_threshold(y_true, proba) -> float:
    """Return the probability threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def compute_metrics(y_true, y_pred, proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }


def stratified_subsample(X, y, size, random_state):
    """Return a stratified subsample of (X, y); pass through if already small."""
    if size <= 0 or size >= len(X):
        return X, y
    X_s, _, y_s, _ = train_test_split(
        X, y, train_size=size, stratify=y, random_state=random_state
    )
    return X_s, y_s


def tune_and_fit(name, tuning_pipeline, factory, grid, X_train, y_train, cv, rs):
    """Grid-search hyper-parameters, choose an F1 threshold, refit on all train.

    Returns (fitted_pipeline, threshold, best_params, cv_score).
    """
    print(f"\n[{name}] grid search ({cv}-fold, scoring=average_precision)...")
    t0 = time.perf_counter()
    search = GridSearchCV(
        tuning_pipeline,
        grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print(
        f"[{name}] best params: {best_params}  "
        f"(cv PR-AUC={search.best_score_:.4f}, {time.perf_counter() - t0:.1f}s)"
    )

    # Choose F1-optimal threshold on a held-out validation split.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=rs
    )
    final = factory(**best_params)
    final.fit(X_fit, y_fit)
    val_proba = final.predict_proba(X_val)[:, 1]
    threshold = best_f1_threshold(y_val, val_proba)
    print(f"[{name}] F1-optimal threshold: {threshold:.3f}")

    # Refit on the entire training sample for the deployable model.
    final.fit(X_train, y_train)
    return final, threshold, best_params, float(search.best_score_)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def make_plots(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    names = list(results.keys())
    palette = {"DecisionTree": "#0386f7", "KNN": "#f7a303"}

    # 1. Metric comparison bar chart.
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(metric_keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, name in enumerate(names):
        vals = [results[name]["metrics"][k] for k in metric_keys]
        ax.bar(x + (i - 0.5) * width, vals, width, label=name,
               color=palette.get(name), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - Evaluation Metrics", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=120)
    plt.close(fig)

    # 2. ROC curves.
    fig, ax = plt.subplots(figsize=(8, 7))
    for name in names:
        roc = results[name]["roc"]
        ax.plot(roc["fpr"], roc["tpr"],
                label=f"{name} (AUC={results[name]['metrics']['roc_auc']:.3f})",
                color=palette.get(name))
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=120)
    plt.close(fig)

    # 3. Confusion matrices.
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        cm = np.array(results[name]["cm"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", cbar=False, ax=ax,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set_title(f"{name} - Confusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved plots to {out_dir}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train heart-disease models.")
    parser.add_argument("--sample", type=int, default=config.TRAIN_SAMPLE_SIZE,
                        help="Max (stratified) training rows. 0 = use all.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plots.")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV.")
    args = parser.parse_args(argv)

    rs = config.RANDOM_STATE
    print(f"scikit-learn {sklearn.__version__} | random_state={rs}")

    df = load_dataset(args.data)
    X, y = split_features_target(df)
    print(f"Dataset: {len(X):,} rows | positive rate: {y.mean():.4f}")

    # Held-out test set from the FULL data (honest evaluation).
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=rs
    )
    # Bounded, stratified training sample (keeps KNN fast + artifact small).
    X_train, y_train = stratified_subsample(X_train_full, y_train_full, args.sample, rs)
    print(f"Train: {len(X_train):,} (sampled from {len(X_train_full):,}) | "
          f"Test: {len(X_test):,}")

    model_specs = [
        ("DecisionTree", decision_tree_pipeline(), decision_tree_pipeline, DT_PARAM_GRID),
        ("KNN", knn_pipeline(n_jobs=1), knn_pipeline, KNN_PARAM_GRID),
    ]

    results: dict[str, dict] = {}
    fitted: dict[str, tuple] = {}
    for name, tuning_pipe, factory, grid in model_specs:
        model, threshold, best_params, cv_score = tune_and_fit(
            name, tuning_pipe, factory, grid, X_train, y_train, config.CV_FOLDS, rs
        )
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        metrics = compute_metrics(y_test, y_pred, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "metrics": metrics,
            "threshold": threshold,
            "best_params": best_params,
            "cv_pr_auc": cv_score,
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "cm": cm.tolist(),
        }
        fitted[name] = (model, threshold, metrics)
        print(f"[{name}] test  acc={metrics['accuracy']:.3f}  "
              f"prec={metrics['precision']:.3f}  rec={metrics['recall']:.3f}  "
              f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

    # Select the best model by F1 (most meaningful under class imbalance).
    best_name = max(results, key=lambda n: results[n]["metrics"]["f1"])
    best_model, best_threshold, best_metrics = fitted[best_name]
    print(f"\n==> Selected best model: {best_name} (F1={best_metrics['f1']:.3f})")

    # Persist artifacts.
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": best_model,
        "threshold": best_threshold,
        "model_name": best_name,
        "raw_features": config.RAW_FEATURES,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "category_values": config.CATEGORY_VALUES,
        "metrics": best_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(X_train)),
        "positive_rate": float(y.mean()),
    }
    dump(bundle, config.MODEL_PATH)
    print(f"Saved model bundle -> {config.MODEL_PATH}")

    metrics_out = {
        "selected_model": best_name,
        "trained_at": bundle["trained_at"],
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "models": {n: {"metrics": r["metrics"], "threshold": r["threshold"],
                       "best_params": r["best_params"], "cv_pr_auc": r["cv_pr_auc"]}
                   for n, r in results.items()},
    }
    (config.ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    # comparison.csv
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    lines = ["model," + ",".join(metric_keys)]
    for n, r in results.items():
        lines.append(n + "," + ",".join(f"{r['metrics'][k]:.4f}" for k in metric_keys))
    (config.ARTIFACTS_DIR / "comparison.csv").write_text("\n".join(lines) + "\n")

    if not args.no_plots:
        make_plots(results, config.PLOTS_DIR)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
