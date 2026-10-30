from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from demos.common import (
    RANDOM_STATE,
    classification_metrics,
)
from demos.demo_bootstrap.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    feature_columns,
)
from model_finder.candidate_result import CandidateRunner, FittedCandidate
from model_finder.candidates import Scoring, scored_candidate
from model_finder.splits import SearchSplits

SCORING = Scoring(TARGET_COLUMN, "val_auc", classification_metrics)
_LOGREG_PARAMS: dict[str, Any] = {"max_iter": 500, "random_state": RANDOM_STATE, "solver": "lbfgs"}


def run_majority_baseline(splits: SearchSplits) -> FittedCandidate:
    """Predict the train-set positive rate for every row (sanity baseline)."""
    positive_rate = float(splits.train[TARGET_COLUMN].mean())
    return scored_candidate(
        splits,
        SCORING,
        lambda rows: np.full(len(rows), positive_rate),
        candidate_id="majority-baseline",
        model_family="constant",
        feature_set="none",
        notes=f"Predict P(y=1)={positive_rate:.4f} for all rows (train prevalence).",
        hyperparameters={"train_positive_rate": positive_rate},
        selected_features=feature_columns(),
        random_seed=None,
    )


def _logreg_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(NUMERIC_COLUMNS),
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(CATEGORICAL_COLUMNS),
            ),
        ]
    )


def _with_engineered(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.assign(
        is_enterprise=(rows["segment"] == "enterprise").astype(int),
        sessions_high=(rows["sessions_7d"] >= 5).astype(int),
    )


def _engineered_numeric_pipeline(
    columns: list[str], *numeric_steps: tuple[str, object]
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                    *numeric_steps,
                                ]
                            ),
                            columns,
                        ),
                    ]
                ),
            ),
            ("estimator", LogisticRegression(**_LOGREG_PARAMS)),
        ]
    )


def _fit_engineered(
    splits: SearchSplits, pipeline: Pipeline, candidate: dict[str, Any]
) -> FittedCandidate:
    columns = candidate["selected_features"]
    pipeline.fit(_with_engineered(splits.train)[columns], splits.train[TARGET_COLUMN])
    return scored_candidate(
        splits,
        SCORING,
        lambda rows: pipeline.predict_proba(_with_engineered(rows)[columns])[:, 1],
        **candidate,
    )


def run_logreg_engineered(splits: SearchSplits) -> FittedCandidate:
    """Drop collinear pages_viewed and noisy days_active. Add threshold features found in EDA."""
    columns = ["sessions_7d", "is_enterprise", "sessions_high"]
    return _fit_engineered(
        splits,
        _engineered_numeric_pipeline(columns),
        dict(
            candidate_id="logreg-engineered",
            model_family="logistic_regression",
            feature_set="engineered",
            notes="Logreg on sessions_7d + is_enterprise + sessions_high. "
            "Drops pages_viewed (collinear) and days_active (noise).",
            hyperparameters=dict(_LOGREG_PARAMS),
            selected_features=columns,
            random_seed=RANDOM_STATE,
        ),
    )


def run_logreg_tiny(splits: SearchSplits) -> FittedCandidate:
    columns = feature_columns()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _logreg_preprocessor()),
            ("estimator", LogisticRegression(**_LOGREG_PARAMS)),
        ]
    )
    pipeline.fit(splits.train[columns], splits.train[TARGET_COLUMN])
    return scored_candidate(
        splits,
        SCORING,
        lambda rows: pipeline.predict_proba(rows[columns])[:, 1],
        candidate_id="logreg-tiny",
        model_family="logistic_regression",
        feature_set="base",
        notes="Logistic regression on numeric + one-hot segment.",
        hyperparameters=dict(_LOGREG_PARAMS),
        selected_features=columns,
        random_seed=RANDOM_STATE,
    )


def run_logreg_poly(splits: SearchSplits) -> FittedCandidate:
    """Test whether degree-2 terms capture the session effect that jumps at 5+ sessions."""
    columns = ["sessions_7d", "is_enterprise"]
    return _fit_engineered(
        splits,
        _engineered_numeric_pipeline(
            columns, ("poly", PolynomialFeatures(degree=2, include_bias=False))
        ),
        dict(
            candidate_id="logreg-poly",
            model_family="logistic_regression",
            feature_set="poly-interactions",
            notes="Logreg with degree-2 polynomial features on sessions_7d + is_enterprise. "
            "Tests nonlinear session effect and enterprise interaction.",
            hyperparameters={**_LOGREG_PARAMS, "C": 1.0, "poly_degree": 2},
            selected_features=columns,
            random_seed=RANDOM_STATE,
        ),
    )


def run_lgbm_conservative(splits: SearchSplits) -> FittedCandidate:
    """Test whether a regularized tree model finds nonlinear effects without engineering."""
    columns = feature_columns()

    def native_categories(rows: pd.DataFrame) -> pd.DataFrame:
        return rows[columns].assign(segment=rows["segment"].astype("category"))

    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 150,
        "num_leaves": 6,
        "max_depth": 3,
        "min_child_samples": 30,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "random_state": RANDOM_STATE,
    }
    model = lgb.LGBMClassifier(**params, verbose=-1)
    model.fit(
        native_categories(splits.train),
        splits.train[TARGET_COLUMN],
        eval_set=[(native_categories(splits.validation), splits.validation[TARGET_COLUMN])],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    best_iteration = int(model.best_iteration_)
    return scored_candidate(
        splits,
        SCORING,
        lambda rows: np.asarray(model.predict_proba(native_categories(rows)))[:, 1],
        candidate_id="lgbm-conservative",
        model_family="lightgbm",
        feature_set="base",
        notes="LightGBM with aggressive regularization (num_leaves=6, max_depth=3, "
        "min_child_samples=30). Same base features as logreg-tiny. "
        f"Early stopped at {best_iteration} iterations.",
        hyperparameters={**params, "best_iteration": best_iteration},
        selected_features=columns,
        random_seed=RANDOM_STATE,
    )


def run_logreg_minimal(splits: SearchSplits) -> FittedCandidate:
    """Test whether the two cleanest signals reduce the train-validation gap."""
    columns = ["sessions_7d", "is_enterprise"]
    return _fit_engineered(
        splits,
        Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("estimator", LogisticRegression(**_LOGREG_PARAMS)),
            ]
        ),
        dict(
            candidate_id="logreg-minimal",
            model_family="logistic_regression",
            feature_set="minimal",
            notes="Logreg on sessions_7d + is_enterprise only. Drops collinear "
            "pages_viewed and noise feature days_active for tighter generalization.",
            hyperparameters=dict(_LOGREG_PARAMS),
            selected_features=columns,
            random_seed=RANDOM_STATE,
        ),
    )


CANDIDATE_RUNNERS: dict[str, CandidateRunner] = {
    "majority-baseline": run_majority_baseline,
    "logreg-tiny": run_logreg_tiny,
    "logreg-engineered": run_logreg_engineered,
    "logreg-minimal": run_logreg_minimal,
    "logreg-poly": run_logreg_poly,
    "lgbm-conservative": run_lgbm_conservative,
}
