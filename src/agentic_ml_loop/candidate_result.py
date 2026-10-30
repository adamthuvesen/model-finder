"""What a candidate returns. Every field here is a rule the type checker or Pydantic enforces."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal, Self

import pandas as pd
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    field_validator,
    model_validator,
)

from agentic_ml_loop.splits import SearchSplits

MINIMUM_BOOTSTRAP_RESAMPLES = 1000

# A metric or hyperparameter name with one of these parts names held-out rows. This stops honest
# code: a score under a neutral name still passes.
_HOLDOUT_TOKENS = frozenset({"test", "final", "unseen", "oot", "oos"})
_HOLDOUT_STEMS = ("holdout", "heldout")
_ALLOWED_HOLDOUT_KEYS = frozenset({"test_size", "final_estimator"})
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATOR = re.compile(r"[^a-z0-9]+")


def names_holdout(key: str) -> bool:
    normalized = _SEPARATOR.sub("_", _CAMEL_BOUNDARY.sub("_", key).lower()).strip("_")
    if normalized in _ALLOWED_HOLDOUT_KEYS:
        return False
    return any(stem in normalized.replace("_", "") for stem in _HOLDOUT_STEMS) or bool(
        _HOLDOUT_TOKENS & set(normalized.split("_"))
    )


def _reject_holdout_keys(values: dict[str, float]) -> dict[str, float]:
    named = sorted(key for key in values if names_holdout(key))
    if named:
        raise ValueError(f"Search metrics must not name holdout rows: {named}")
    return values


Direction = Literal["minimize", "maximize"]
MetricValues = Annotated[dict[str, FiniteFloat], Field(min_length=1)]
SearchMetricValues = Annotated[MetricValues, AfterValidator(_reject_holdout_keys)]
StrictSeed = Annotated[int, Field(strict=True)]
NonEmpty = Annotated[str, Field(min_length=1, pattern=r"\S")]


class SearchMetrics(BaseModel):
    """Allowlist of split keys. Any other key, such as ``final_holdout``, fails validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    train: SearchMetricValues
    validation: SearchMetricValues


def canonical_objective_metric(metric: str) -> str:
    return (
        "val_" + metric.removeprefix("validation_") if metric.startswith("validation_") else metric
    )


def metric_key(objective_metric: str) -> str:
    return objective_metric.removeprefix("validation_").removeprefix("val_")


def train_validation_gap(
    metrics: SearchMetrics, objective_metric: str, direction: Direction = "maximize"
) -> float:
    """Return the gap in the direction of overfit: a positive gap favors train."""
    key = metric_key(objective_metric)
    gap = metrics.train[key] - metrics.validation[key]
    return gap if direction == "maximize" else -gap


def _check_json_value(value: JsonValue, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_json_value(item, f"{path}[{index}]")
    if isinstance(value, dict):
        for key, item in value.items():
            if names_holdout(key):
                raise ValueError(f"{path}.{key} must not name holdout rows")
            _check_json_value(item, f"{path}.{key}")


class CandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: NonEmpty
    model_family: NonEmpty
    feature_set: NonEmpty
    objective_metric: NonEmpty
    objective_score: FiniteFloat
    split_strategy: NonEmpty
    notes: str
    metrics: SearchMetrics
    hyperparameters: dict[str, JsonValue]
    selected_features: list[str]
    random_seed: StrictSeed | None
    bootstrap_count: Annotated[int, Field(strict=True, ge=MINIMUM_BOOTSTRAP_RESAMPLES)] | None
    train_validation_gap: FiniteFloat
    objective_direction: Direction = "maximize"
    status: Literal["completed"] = "completed"

    @model_validator(mode="after")
    def _objective_is_a_validation_metric(self) -> Self:
        if not self.objective_metric.startswith(("val_", "validation_")):
            raise ValueError("objective_metric must start with val_ or validation_")
        if names_holdout(metric_key(self.objective_metric)):
            raise ValueError("objective_metric must not name holdout rows")
        key = metric_key(self.objective_metric)
        if key not in self.metrics.train or key not in self.metrics.validation:
            raise ValueError(f"metrics.train and metrics.validation must both report {key}")
        if not math.isclose(
            self.objective_score, self.metrics.validation[key], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"objective_score differs from metrics.validation.{key}")
        return self

    @field_validator("hyperparameters")
    @classmethod
    def _hyperparameters_are_finite_and_search_only(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        _check_json_value(value, "hyperparameters")
        return value

    @model_validator(mode="after")
    def _gap_matches_metrics(self) -> Self:
        expected = train_validation_gap(
            self.metrics, self.objective_metric, self.objective_direction
        )
        if not math.isclose(self.train_validation_gap, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"train_validation_gap must be {expected}")
        return self

    @model_validator(mode="after")
    def _interval_declares_bootstrap_count(self) -> Self:
        has_interval = any(key.endswith("_ci_95") for key in self.hyperparameters)
        if has_interval != (self.bootstrap_count is not None):
            raise ValueError("A *_ci_95 hyperparameter and bootstrap_count require each other")
        return self


@dataclass(frozen=True)
class FittedCandidate:
    """A fitted model. ``score_rows`` is how the engine reaches holdout rows after a freeze."""

    result: CandidateResult
    score_rows: Callable[[pd.DataFrame], dict[str, float]]


CandidateRunner = Callable[[SearchSplits], FittedCandidate]
