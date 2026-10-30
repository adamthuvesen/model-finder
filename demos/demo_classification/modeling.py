from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import JsonValue
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from agentic_ml_loop.candidate_result import CandidateRunner, FittedCandidate
from agentic_ml_loop.splits import SearchSplits
from demos.common import (
    RANDOM_STATE,
    DemoCandidate,
    DemoScoring,
    classification_metrics,
    fitted_candidate,
)
from demos.demo_classification.data import (
    BASE_NUMERIC_COLUMNS,
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    TARGET_COLUMN,
    base_feature_columns,
    engineered_feature_columns,
)

SCORING = DemoScoring(TARGET_COLUMN, "val_auc", classification_metrics)


@dataclass(frozen=True)
class PipelineModelSpec:
    candidate_id: str
    model_family: str
    feature_set: str
    estimator: Any
    notes: str
    hyperparameters: dict[str, JsonValue]


def _preprocessor(feature_set: str) -> ColumnTransformer:
    if feature_set not in ("base", "engineered"):
        raise ValueError(f"Unknown feature_set: {feature_set!r}. Must be 'base' or 'engineered'.")
    numeric_features = list(BASE_NUMERIC_COLUMNS)
    if feature_set == "engineered":
        numeric_features += ENGINEERED_NUMERIC_COLUMNS
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
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
            (
                "boolean",
                "passthrough",
                BOOLEAN_COLUMNS,
            ),
        ]
    )


def _fit_pipeline_model(spec: PipelineModelSpec, splits: SearchSplits) -> FittedCandidate:
    selected_features = (
        engineered_feature_columns() if spec.feature_set == "engineered" else base_feature_columns()
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _preprocessor(spec.feature_set)),
            ("estimator", spec.estimator),
        ]
    )
    pipeline.fit(splits.train[selected_features], splits.train[TARGET_COLUMN])
    return fitted_candidate(
        splits,
        SCORING,
        DemoCandidate(
            candidate_id=spec.candidate_id,
            model_family=spec.model_family,
            feature_set=spec.feature_set,
            notes=spec.notes,
            hyperparameters=spec.hyperparameters,
            selected_features=selected_features,
        ),
        lambda rows: pipeline.predict_proba(rows[selected_features])[:, 1],
    )


def run_rule_baseline(splits: SearchSplits) -> FittedCandidate:
    def score(frame: pd.DataFrame) -> np.ndarray:
        raw = (
            0.35 * (frame["usage_growth_90d"] > 0.12).astype(float)
            + 0.20 * (frame["feature_adoption_count"] >= 4).astype(float)
            + 0.15 * (frame["seat_utilization"] > 0.22).astype(float)
            + 0.15 * frame["has_exec_sponsor"].astype(float)
            + 0.10 * frame["previous_expansion"].astype(float)
            + 0.05 * (frame["plan_tier"].isin(["pro", "business"])).astype(float)
        )
        return np.clip(0.05 + 0.90 * raw, 0.05, 0.95)

    return fitted_candidate(
        splits,
        SCORING,
        DemoCandidate(
            candidate_id="rule-baseline",
            model_family="rule_based",
            feature_set="heuristic",
            notes="Thresholded heuristic over usage growth, feature adoption, and sponsor signals.",
            hyperparameters={},
            selected_features=[
                "usage_growth_90d",
                "feature_adoption_count",
                "seat_utilization",
                "has_exec_sponsor",
                "previous_expansion",
                "plan_tier",
            ],
            random_seed=None,
        ),
        score,
    )


def run_logreg_basic(splits: SearchSplits) -> FittedCandidate:
    return _fit_pipeline_model(
        PipelineModelSpec(
            candidate_id="logreg-basic",
            model_family="logistic_regression",
            feature_set="base",
            estimator=LogisticRegression(max_iter=500, random_state=RANDOM_STATE, solver="lbfgs"),
            notes="Logistic regression over the base feature set.",
            hyperparameters={
                "max_iter": 500,
                "random_state": RANDOM_STATE,
                "solver": "lbfgs",
            },
        ),
        splits,
    )


def run_logreg_engineered(splits: SearchSplits) -> FittedCandidate:
    return _fit_pipeline_model(
        PipelineModelSpec(
            candidate_id="logreg-engineered",
            model_family="logistic_regression",
            feature_set="engineered",
            estimator=LogisticRegression(max_iter=800, random_state=RANDOM_STATE, solver="lbfgs"),
            notes="Logistic regression with engineered utilization, support intensity, and momentum features.",
            hyperparameters={
                "max_iter": 800,
                "random_state": RANDOM_STATE,
                "solver": "lbfgs",
            },
        ),
        splits,
    )


def run_xgb_base(splits: SearchSplits) -> FittedCandidate:
    estimator = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=2.0,
        reg_alpha=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    return _fit_pipeline_model(
        PipelineModelSpec(
            candidate_id="xgb-base",
            model_family="xgboost",
            feature_set="base",
            estimator=estimator,
            notes="XGBoost on base features with conservative regularization for small dataset.",
            hyperparameters={
                "n_estimators": 150,
                "max_depth": 3,
                "learning_rate": 0.08,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "min_child_weight": 5,
                "reg_lambda": 2.0,
                "reg_alpha": 0.1,
                "random_state": RANDOM_STATE,
                "eval_metric": "logloss",
            },
        ),
        splits,
    )


def run_xgb_basic(splits: SearchSplits) -> FittedCandidate:
    estimator = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    return _fit_pipeline_model(
        PipelineModelSpec(
            candidate_id="xgb-basic",
            model_family="xgboost",
            feature_set="engineered",
            estimator=estimator,
            notes="XGBoost baseline over engineered features.",
            hyperparameters={
                "n_estimators": 180,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": RANDOM_STATE,
                "eval_metric": "logloss",
            },
        ),
        splits,
    )


def run_xgb_tuned_light(splits: SearchSplits) -> FittedCandidate:
    search_space = [
        {
            "n_estimators": 120,
            "max_depth": 3,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        {
            "n_estimators": 180,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
        },
        {
            "n_estimators": 220,
            "max_depth": 5,
            "learning_rate": 0.04,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
        },
        {
            "n_estimators": 140,
            "max_depth": 4,
            "learning_rate": 0.07,
            "subsample": 1.0,
            "colsample_bytree": 0.9,
        },
    ]

    selected_features = engineered_feature_columns()
    train_x = splits.train[selected_features]
    train_y = splits.train[TARGET_COLUMN]
    validation_x = splits.validation[selected_features]
    validation_y = splits.validation[TARGET_COLUMN]

    best_auc = float("-inf")
    best_pipeline: Pipeline | None = None
    best_params: dict[str, JsonValue] | None = None
    for params in search_space:
        estimator = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", **params)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _preprocessor("engineered")),
                ("estimator", estimator),
            ]
        )
        pipeline.fit(train_x, train_y)
        validation_auc = roc_auc_score(validation_y, pipeline.predict_proba(validation_x)[:, 1])
        if validation_auc > best_auc:
            best_auc = float(validation_auc)
            best_pipeline = pipeline
            best_params = {**params, "random_state": RANDOM_STATE, "eval_metric": "logloss"}

    if best_pipeline is None or best_params is None:
        raise RuntimeError("No valid pipeline found during tuning search.")
    chosen_pipeline = best_pipeline
    return fitted_candidate(
        splits,
        SCORING,
        DemoCandidate(
            candidate_id="xgb-tuned-light",
            model_family="xgboost",
            feature_set="engineered",
            notes="Light manual sweep across four XGBoost configs, selecting by validation AUC.",
            hyperparameters=best_params,
            selected_features=selected_features,
        ),
        lambda rows: chosen_pipeline.predict_proba(rows[selected_features])[:, 1],
    )


CANDIDATE_RUNNERS: dict[str, CandidateRunner] = {
    "rule-baseline": run_rule_baseline,
    "logreg-basic": run_logreg_basic,
    "logreg-engineered": run_logreg_engineered,
    "xgb-base": run_xgb_base,
    "xgb-basic": run_xgb_basic,
    "xgb-tuned-light": run_xgb_tuned_light,
}
