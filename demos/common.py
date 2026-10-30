"""Scoring and result assembly shared by the demos."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import JsonValue
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from agentic_ml_loop.candidate_result import (
    CandidateResult,
    FittedCandidate,
    SearchMetrics,
    metric_key,
    train_validation_gap,
)
from agentic_ml_loop.splits import SearchSplits

CODE_ROOT = Path(__file__).resolve().parent
SPLIT_STRATEGY = "time_split_60_20_20"
RANDOM_STATE = 42

Predict = Callable[[pd.DataFrame], np.ndarray]
Evaluate = Callable[[pd.Series, np.ndarray], dict[str, float]]


def classification_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "recall": float(recall_score(y_true, predictions, zero_division=0)),  # pyright: ignore[reportArgumentType]
    }


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
    }


@dataclass(frozen=True)
class DemoScoring:
    target_column: str
    objective_metric: str
    evaluate: Evaluate


@dataclass(frozen=True)
class DemoCandidate:
    candidate_id: str
    model_family: str
    feature_set: str
    notes: str
    hyperparameters: dict[str, JsonValue]
    selected_features: list[str]
    random_seed: int | None = RANDOM_STATE


def fitted_candidate(
    splits: SearchSplits, scoring: DemoScoring, candidate: DemoCandidate, predict: Predict
) -> FittedCandidate:
    """Score a fitted model on search rows. The engine calls ``score_rows`` after a freeze."""

    def score_rows(rows: pd.DataFrame) -> dict[str, float]:
        return scoring.evaluate(rows[scoring.target_column], predict(rows))

    metrics = SearchMetrics(
        train=score_rows(splits.train), validation=score_rows(splits.validation)
    )
    return FittedCandidate(
        result=CandidateResult(
            candidate_id=candidate.candidate_id,
            model_family=candidate.model_family,
            feature_set=candidate.feature_set,
            objective_metric=scoring.objective_metric,
            objective_score=metrics.validation[metric_key(scoring.objective_metric)],
            split_strategy=SPLIT_STRATEGY,
            notes=candidate.notes,
            metrics=metrics,
            hyperparameters=candidate.hyperparameters,
            selected_features=candidate.selected_features,
            random_seed=candidate.random_seed,
            bootstrap_count=None,
            train_validation_gap=train_validation_gap(metrics, scoring.objective_metric),
        ),
        score_rows=score_rows,
    )
