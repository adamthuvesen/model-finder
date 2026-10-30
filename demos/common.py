"""Metrics shared by the demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
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

CODE_ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42


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
