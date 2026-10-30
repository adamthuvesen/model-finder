"""Small scores with no extra dependency, for a first leaderboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _labels(target: pd.Series) -> np.ndarray:
    """Refuse missing labels, because any score over them would look real and be wrong."""
    missing = int(target.isna().sum())
    if missing:
        raise ValueError(
            f"column {target.name} has {missing} missing values in these rows. "
            f"Fix: drop those rows in rows_loader, with `.dropna(subset=[{str(target.name)!r}])`"
        )
    return target.to_numpy(dtype=float)


def classification_scores(target: pd.Series, scores: np.ndarray) -> dict[str, float]:
    """Return the area under the ROC curve, from the rank-sum statistic."""
    labels = _labels(target)
    if not set(np.unique(labels)) <= {0.0, 1.0}:
        raise ValueError(
            f"column {target.name} has values that are not 0 and 1. "
            "Fix: encode it as 0 and 1, or use `Scoring.regression` in candidates.py"
        )
    positives = labels == 1
    positive_count, negative_count = int(positives.sum()), int((~positives).sum())
    if not positive_count or not negative_count:
        raise ValueError(
            f"column {target.name} has only one class in these rows. "
            "Fix: use more rows, or use `Scoring.regression` in candidates.py"
        )
    ranks = pd.Series(scores).rank().to_numpy()
    rank_sum = float(ranks[positives].sum())
    return {
        "auc": (rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    }


def regression_scores(target: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    errors = _labels(target) - np.asarray(predictions, dtype=float)
    return {"rmse": float(np.sqrt((errors**2).mean())), "mae": float(np.abs(errors).mean())}
