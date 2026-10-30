"""Write a candidate as one function. The engine scores it and records everything else."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model_finder.candidate_result import (
    CandidateResult,
    CandidateRunner,
    Direction,
    FittedCandidate,
    SearchMetrics,
    metric_key,
    names_holdout,
    run_candidate,
    train_validation_gap,
)
from model_finder.metrics import classification_scores, regression_scores
from model_finder.splits import SearchSplits

Predict = Callable[[pd.DataFrame], np.ndarray]
Evaluate = Callable[[pd.Series, np.ndarray], dict[str, float]]
Fit = Callable[[SearchSplits], Predict]


@dataclass(frozen=True)
class Scoring:
    """How every candidate in an experiment is scored."""

    target_column: str
    objective_metric: str
    evaluate: Evaluate
    objective_direction: Direction = "maximize"

    @classmethod
    def classification(cls, target_column: str) -> Scoring:
        return cls(target_column, "val_auc", classification_scores)

    @classmethod
    def regression(cls, target_column: str) -> Scoring:
        return cls(target_column, "val_rmse", regression_scores, "minimize")


def code_location(function: object) -> str:
    """Name where a function is defined, for a message that points at the line to change."""
    code = getattr(function, "__code__", None)
    if code is None:
        return "candidates"
    path = Path(code.co_filename)
    shown = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
    return f"{shown}:{code.co_firstlineno}"


def _checked_predictions(predict: Predict, rows: pd.DataFrame) -> np.ndarray:
    predictions = np.asarray(predict(rows))
    if predictions.ndim != 1:
        raise ValueError(
            f"{code_location(predict)}: predict returned shape {predictions.shape}, not one value "
            "per row. Fix: return one column, such as `model.predict_proba(rows)[:, 1]`"
        )
    if len(predictions) != len(rows):
        raise ValueError(
            f"{code_location(predict)}: predict returned {len(predictions)} predictions for "
            f"{len(rows)} rows. Fix: return one value for each row it receives"
        )
    return predictions


def scored_candidate(
    splits: SearchSplits,
    scoring: Scoring,
    predict: Predict,
    *,
    candidate_id: str,
    **metadata: object,
) -> FittedCandidate:
    """Score a fitted model. ``predict`` never receives the target, so it cannot leak into a score."""

    def score_rows(rows: pd.DataFrame) -> dict[str, float]:
        features = rows.drop(columns=[scoring.target_column])
        return scoring.evaluate(
            rows[scoring.target_column], _checked_predictions(predict, features)
        )

    train, validation = score_rows(splits.train), score_rows(splits.validation)
    named = sorted(name for name in {*train, *validation} if names_holdout(name))
    if named:
        raise ValueError(
            f"{code_location(scoring.evaluate)}: metric {', '.join(named)} looks like a holdout "
            "score. Fix: remove it from this function, and use evaluate-final for holdout scores"
        )
    metrics = SearchMetrics(train=train, validation=validation)
    key = metric_key(scoring.objective_metric)
    if key not in metrics.validation:
        valid = ", ".join(f"val_{name}" for name in metrics.validation)
        raise ValueError(
            f"objective {scoring.objective_metric} is not among the metrics. "
            f"Fix: use one of: {valid}"
        )
    result = CandidateResult.model_validate(
        {
            "candidate_id": candidate_id,
            "objective_metric": scoring.objective_metric,
            "objective_direction": scoring.objective_direction,
            "objective_score": metrics.validation[key],
            "metrics": metrics,
            "train_validation_gap": train_validation_gap(
                metrics, scoring.objective_metric, scoring.objective_direction
            ),
            **metadata,
        }
    )
    return FittedCandidate(result=result, score_rows=score_rows)


class Candidates(dict[str, CandidateRunner]):
    """Candidates by name. ``@candidates.add`` turns a function into a runner."""

    def __init__(self, scoring: Scoring) -> None:
        super().__init__()
        self.scoring = scoring
        self._locations: dict[str, str] = {}

    def add(self, fit: Fit) -> Fit:
        """Register ``fit``. Its name is the candidate id. Its docstring is the notes."""
        candidate_id = fit.__name__.replace("_", "-")
        location = code_location(fit)
        if candidate_id in self:
            raise ValueError(
                f"{location}: candidate {candidate_id} already exists at "
                f"{self._locations[candidate_id]}. Fix: rename one of them"
            )

        def runner(splits: SearchSplits) -> FittedCandidate:
            def fit_and_score(given: SearchSplits) -> FittedCandidate:
                predict = fit(given)
                if not callable(predict):
                    raise TypeError(
                        f"{location}: {fit.__name__} did not return a function that predicts. "
                        "Fix: return one, such as `return model.predict`"
                    )
                return scored_candidate(
                    given,
                    self.scoring,
                    predict,
                    candidate_id=candidate_id,
                    notes=(fit.__doc__ or "").strip(),
                )

            return run_candidate(fit_and_score, splits)

        self[candidate_id] = runner
        self._locations[candidate_id] = location
        return fit
