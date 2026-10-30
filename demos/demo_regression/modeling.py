from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import JsonValue
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from agentic_ml_loop.candidate_result import CandidateRunner, FittedCandidate
from agentic_ml_loop.splits import SearchSplits
from demos.common import (
    RANDOM_STATE,
    DemoCandidate,
    DemoScoring,
    fitted_candidate,
    regression_metrics,
)
from demos.demo_regression.data import (
    BASE_NUMERIC_COLUMNS,
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    TARGET_COLUMN,
    base_feature_columns,
    engineered_feature_columns,
)

logger = logging.getLogger(__name__)

SCORING = DemoScoring(TARGET_COLUMN, "val_r2", regression_metrics)


@dataclass(frozen=True)
class RegressionModelSpec:
    candidate_id: str
    model_family: str
    feature_set: str
    estimator: Any
    notes: str
    hyperparameters: dict[str, JsonValue]
    preprocessor: ColumnTransformer


def _regression_preprocessor(feature_set: str) -> ColumnTransformer:
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


def _xgb_preprocessor(feature_set: str) -> ColumnTransformer:
    if feature_set not in ("base", "engineered"):
        raise ValueError(f"Unknown feature_set: {feature_set!r}. Must be 'base' or 'engineered'.")
    numeric_features = list(BASE_NUMERIC_COLUMNS)
    if feature_set == "engineered":
        numeric_features += ENGINEERED_NUMERIC_COLUMNS
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
            ("boolean", "passthrough", BOOLEAN_COLUMNS),
        ]
    )


def _demo_candidate(spec: RegressionModelSpec, selected_features: list[str]) -> DemoCandidate:
    return DemoCandidate(
        candidate_id=spec.candidate_id,
        model_family=spec.model_family,
        feature_set=spec.feature_set,
        notes=spec.notes,
        hyperparameters=spec.hyperparameters,
        selected_features=selected_features,
    )


def _spec_features(spec: RegressionModelSpec) -> list[str]:
    return (
        engineered_feature_columns() if spec.feature_set == "engineered" else base_feature_columns()
    )


def _fit_log_transform_model(spec: RegressionModelSpec, splits: SearchSplits) -> FittedCandidate:
    """Fit a pipeline with log(1+y) target transform; evaluate predictions on original scale."""
    selected_features = _spec_features(spec)
    wrapped = TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[("preprocessor", spec.preprocessor), ("estimator", spec.estimator)]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    wrapped.fit(splits.train[selected_features], splits.train[TARGET_COLUMN])
    return fitted_candidate(
        splits,
        SCORING,
        _demo_candidate(spec, selected_features),
        lambda rows: wrapped.predict(rows[selected_features]),
    )


def _fit_pipeline_model(spec: RegressionModelSpec, splits: SearchSplits) -> FittedCandidate:
    selected_features = _spec_features(spec)
    pipeline = Pipeline(steps=[("preprocessor", spec.preprocessor), ("estimator", spec.estimator)])
    pipeline.fit(splits.train[selected_features], splits.train[TARGET_COLUMN])
    return fitted_candidate(
        splits,
        SCORING,
        _demo_candidate(spec, selected_features),
        lambda rows: pipeline.predict(rows[selected_features]),
    )


def run_rule_baseline(splits: SearchSplits) -> FittedCandidate:
    rule_features = [
        "usage_growth_90d",
        "feature_adoption_count",
        "champion_engagement_score",
        "seat_utilization",
        "has_exec_sponsor",
        "multi_product",
    ]
    weights = np.array([0.24, 0.16, 0.18, 0.20, 0.12, 0.10], dtype=float)

    def raw_score(frame: pd.DataFrame) -> np.ndarray:
        result = np.zeros(len(frame), dtype=float)
        for feature, weight in zip(rule_features, weights, strict=True):
            values = frame[feature].astype(float).to_numpy()
            ref_values = np.sort(splits.train[feature].astype(float).to_numpy())
            ranks = np.searchsorted(ref_values, values) / max(len(ref_values), 1)
            result += weight * ranks
        return result

    train_raw = raw_score(splits.train)
    design = np.column_stack([np.ones_like(train_raw), train_raw])
    intercept, slope = np.linalg.lstsq(design, splits.train[TARGET_COLUMN].to_numpy(), rcond=None)[
        0
    ]
    return fitted_candidate(
        splits,
        SCORING,
        DemoCandidate(
            candidate_id="rule-baseline",
            model_family="rule_based",
            feature_set="heuristic",
            notes="Train-calibrated heuristic over growth, adoption, champion strength, and seat utilization.",
            hyperparameters={
                "weights": dict(zip(rule_features, weights.tolist(), strict=True)),
                "calibration": {"intercept": float(intercept), "slope": float(slope)},
            },
            selected_features=rule_features,
            random_seed=None,
        ),
        lambda rows: intercept + slope * raw_score(rows),
    )


def run_ridge_basic(splits: SearchSplits) -> FittedCandidate:
    return _fit_pipeline_model(
        RegressionModelSpec(
            candidate_id="ridge-basic",
            model_family="ridge_regression",
            feature_set="base",
            estimator=Ridge(alpha=2.0),
            notes="Ridge regression over base features with train-fitted scaling and one-hot categoricals.",
            hyperparameters={
                "alpha": 2.0,
            },
            preprocessor=_regression_preprocessor("base"),
        ),
        splits,
    )


def run_ridge_engineered(splits: SearchSplits) -> FittedCandidate:
    return _fit_pipeline_model(
        RegressionModelSpec(
            candidate_id="ridge-engineered",
            model_family="ridge_regression",
            feature_set="engineered",
            estimator=Ridge(alpha=3.0),
            notes="Ridge regression with engineered utilization, burden, and capacity features.",
            hyperparameters={
                "alpha": 3.0,
            },
            preprocessor=_regression_preprocessor("engineered"),
        ),
        splits,
    )


def run_xgb_basic(splits: SearchSplits) -> FittedCandidate:
    estimator = XGBRegressor(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=12,
        reg_alpha=0.4,
        reg_lambda=4.0,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        eval_metric="rmse",
    )
    return _fit_pipeline_model(
        RegressionModelSpec(
            candidate_id="xgb-basic",
            model_family="xgboost_regressor",
            feature_set="engineered",
            estimator=estimator,
            notes="Regularized XGBoost regressor over base plus engineered regression features.",
            hyperparameters={
                "n_estimators": 260,
                "max_depth": 4,
                "learning_rate": 0.04,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 12,
                "reg_alpha": 0.4,
                "reg_lambda": 4.0,
                "random_state": RANDOM_STATE,
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
            },
            preprocessor=_xgb_preprocessor("engineered"),
        ),
        splits,
    )


def run_ridge_log(splits: SearchSplits) -> FittedCandidate:
    """Ridge on base features with log(1+y) target transform; evaluate on original scale."""
    return _fit_log_transform_model(
        RegressionModelSpec(
            candidate_id="ridge-log",
            model_family="ridge_regression",
            feature_set="base",
            estimator=Ridge(alpha=2.0),
            notes="Ridge on base features with log(1+y) target transform; back-transform to evaluate on original scale.",
            hyperparameters={
                "alpha": 2.0,
                "target_transform": "log1p",
            },
            preprocessor=_regression_preprocessor("base"),
        ),
        splits,
    )


def run_xgb_log(splits: SearchSplits) -> FittedCandidate:
    """XGBoost with log(1+y) target transform; tests H1 — log transform reduces right-tail MSE dominance."""
    estimator = XGBRegressor(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=12,
        reg_alpha=0.4,
        reg_lambda=4.0,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        eval_metric="rmse",
    )
    return _fit_log_transform_model(
        RegressionModelSpec(
            candidate_id="xgb-log",
            model_family="xgboost_regressor",
            feature_set="engineered",
            estimator=estimator,
            notes="XGBoost with log(1+y) target transform; same architecture as xgb-basic, tests H1 for tree models.",
            hyperparameters={
                "n_estimators": 260,
                "max_depth": 4,
                "learning_rate": 0.04,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 12,
                "reg_alpha": 0.4,
                "reg_lambda": 4.0,
                "random_state": RANDOM_STATE,
                "target_transform": "log1p",
            },
            preprocessor=_xgb_preprocessor("engineered"),
        ),
        splits,
    )


def run_hurdle_logistic_ridge(splits: SearchSplits) -> FittedCandidate:
    """Two-stage hurdle: P(positive) from a classifier times E[amount | positive] from ridge."""
    selected_features = base_feature_columns()
    preprocessor = _regression_preprocessor("base")
    train_x = preprocessor.fit_transform(splits.train[selected_features])

    positive_mask = splits.train[TARGET_COLUMN] > 0
    stage1 = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    stage1.fit(train_x, positive_mask.astype(int))
    stage2 = Ridge(alpha=2.0)
    stage2.fit(train_x[positive_mask.to_numpy()], splits.train.loc[positive_mask, TARGET_COLUMN])

    def predict(rows: pd.DataFrame) -> np.ndarray:
        transformed = preprocessor.transform(rows[selected_features])
        prob_positive = stage1.predict_proba(transformed)[:, 1]
        return prob_positive * np.maximum(stage2.predict(transformed), 0.0)

    return fitted_candidate(
        splits,
        SCORING,
        DemoCandidate(
            candidate_id="hurdle-logistic-ridge",
            model_family="two_stage_hurdle",
            feature_set="base",
            notes=(
                "Two-stage hurdle: logistic(base) for P(positive), "
                "ridge(base) on positive-only train subset for E[amount|positive]. "
                "Soft prediction: P(positive) * E[amount|positive]."
            ),
            hyperparameters={
                "stage1": {"model": "LogisticRegression", "C": 1.0},
                "stage2": {"model": "Ridge", "alpha": 2.0},
                "prediction_mode": "soft_probability",
            },
            selected_features=selected_features,
        ),
        predict,
    )


def run_xgb_tweedie(splits: SearchSplits) -> FittedCandidate:
    """XGBoost with Tweedie loss (power=1.5) — native single-model treatment for
    zero-inflated, right-skewed targets. Avoids log transform while changing the
    loss surface to handle zeros and positives as a compound Poisson-Gamma process.
    """
    estimator = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=12,
        reg_alpha=0.4,
        reg_lambda=4.0,
        random_state=RANDOM_STATE,
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
    )
    return _fit_pipeline_model(
        RegressionModelSpec(
            candidate_id="xgb-tweedie",
            model_family="xgboost_tweedie",
            feature_set="base",
            estimator=estimator,
            notes=(
                "XGBoost with Tweedie loss (power=1.5) for zero-inflated right-skewed target. "
                "Single-model alternative to two-stage hurdle; same regularization as xgb-basic."
            ),
            hyperparameters={
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.04,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 12,
                "reg_alpha": 0.4,
                "reg_lambda": 4.0,
                "random_state": RANDOM_STATE,
                "objective": "reg:tweedie",
                "tweedie_variance_power": 1.5,
            },
            preprocessor=_xgb_preprocessor("base"),
        ),
        splits,
    )


def run_ridge_scale_normalized(splits: SearchSplits) -> FittedCandidate:
    """Predict expansion rate, then back-multiply. Tests for signal independent of account size."""
    selected_features = base_feature_columns()
    revenue_col = "contract_value_t0"
    train_revenue = splits.train[revenue_col].to_numpy()
    floored = int((train_revenue < 1.0).sum())
    if floored > 0:
        logger.warning(
            "run_ridge_scale_normalized: %d train row(s) have %s < 1.0; flooring to 1.0 for rate computation",
            floored,
            revenue_col,
        )
    train_rate = splits.train[TARGET_COLUMN].to_numpy() / np.maximum(train_revenue, 1.0)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _regression_preprocessor("base")),
            ("estimator", Ridge(alpha=2.0)),
        ]
    )
    pipeline.fit(splits.train[selected_features], train_rate)

    def predict(rows: pd.DataFrame) -> np.ndarray:
        rate = pipeline.predict(rows[selected_features])
        return np.maximum(rate * rows[revenue_col].to_numpy(), 0.0)

    return fitted_candidate(
        splits,
        SCORING,
        DemoCandidate(
            candidate_id="ridge-scale-norm",
            model_family="ridge_regression",
            feature_set="base_normalized",
            notes=(
                "H2: predicts expansion_rate = target / contract_value_t0, back-multiplied. "
                "Diagnostic for scale confound: if R2 drops significantly, raw R2 is mostly scale."
            ),
            hyperparameters={
                "alpha": 2.0,
                "target_transform": "rate_normalization",
                "normalization_feature": revenue_col,
            },
            selected_features=selected_features,
        ),
        predict,
    )


CANDIDATE_RUNNERS: dict[str, CandidateRunner] = {
    "rule-baseline": run_rule_baseline,
    "ridge-basic": run_ridge_basic,
    "ridge-engineered": run_ridge_engineered,
    "xgb-basic": run_xgb_basic,
    "ridge-log": run_ridge_log,
    "xgb-log": run_xgb_log,
    "hurdle-logistic-ridge": run_hurdle_logistic_ridge,
    "xgb-tweedie": run_xgb_tweedie,
    "ridge-scale-norm": run_ridge_scale_normalized,
}
