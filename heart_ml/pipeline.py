"""Model pipelines.

Every estimator is wrapped in a single sklearn ``Pipeline`` that performs, in
order: feature engineering -> preprocessing (scaling + one-hot encoding) ->
classifier. Because preprocessing lives inside the pipeline and is only ``fit``
on the training fold, there is no data leakage into the test set (unlike the
original notebook, which called ``scaler.fit_transform`` on the test data).
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config
from .features import FeatureEngineer


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric+engineered columns, one-hot encode categoricals.

    ``handle_unknown='ignore'`` makes serving robust: if a request ever carries
    an unseen category the encoder emits all-zeros instead of crashing.
    """
    numeric = config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("clf", classifier),
        ]
    )


def decision_tree_pipeline(**params) -> Pipeline:
    """Decision Tree with balanced class weights (handles 8.5% positive rate)."""
    defaults = dict(
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
    )
    defaults.update(params)
    return build_pipeline(DecisionTreeClassifier(**defaults))


def knn_pipeline(**params) -> Pipeline:
    """K-Nearest Neighbors. ``n_jobs=-1`` parallelizes neighbor search."""
    defaults = dict(n_jobs=-1)
    defaults.update(params)
    return build_pipeline(KNeighborsClassifier(**defaults))


# Hyper-parameter grids used by train.py (kept modest to stay fast on KNN).
DT_PARAM_GRID = {
    "clf__max_depth": [6, 10, 16, None],
    "clf__min_samples_leaf": [20, 50, 100],
    "clf__criterion": ["gini", "entropy"],
}

KNN_PARAM_GRID = {
    "clf__n_neighbors": [15, 31, 51],
    "clf__weights": ["distance"],
}
