from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_ml_loop.candidate_result import CandidateResult, SearchMetrics, train_validation_gap
from tests.conftest import candidate_result


def _payload(**overrides: object) -> dict[str, object]:
    return {**candidate_result().model_dump(), **overrides}


@pytest.mark.parametrize("split", ["test", "test_q4", "holdout", "final_holdout", "validation_2"])
def test_split_keys_are_an_allowlist(split: str) -> None:
    metrics = {"train": {"auc": 0.7}, "validation": {"auc": 0.7}, split: {"auc": 0.9}}
    with pytest.raises(ValidationError, match=split):
        CandidateResult.model_validate(_payload(metrics=metrics))


def test_a_quick_candidate_needs_only_its_score() -> None:
    result = CandidateResult(
        candidate_id="quick",
        objective_metric="val_auc",
        objective_score=0.7,
        metrics=SearchMetrics(validation={"auc": 0.7}),
    )
    assert result.random_seed is None
    assert result.bootstrap_count is None
    assert result.train_validation_gap is None


def test_gap_must_match_the_metrics() -> None:
    with pytest.raises(ValidationError, match="train_validation_gap"):
        CandidateResult.model_validate(_payload(train_validation_gap=0.2))


def test_gap_points_toward_overfit_for_a_minimized_metric() -> None:
    metrics = SearchMetrics(train={"rmse": 1.0}, validation={"rmse": 3.0})
    assert train_validation_gap(metrics, "val_rmse", "minimize") == 2.0


def test_interval_requires_a_bootstrap_count() -> None:
    interval = {"val_auc_ci_95": [0.6, 0.8]}
    with pytest.raises(ValidationError, match="requires bootstrap_count"):
        CandidateResult.model_validate(_payload(hyperparameters=interval))
    CandidateResult.model_validate(_payload(hyperparameters=interval, bootstrap_count=200))


def test_objective_must_be_the_validation_metric() -> None:
    with pytest.raises(ValidationError, match="val_ or validation_"):
        CandidateResult.model_validate(_payload(objective_metric="test_auc"))
    with pytest.raises(ValidationError, match="differs"):
        CandidateResult.model_validate(_payload(objective_score=0.1))


def test_non_finite_metric_is_rejected() -> None:
    metrics = {"train": {"auc": 0.7}, "validation": {"auc": float("nan")}}
    with pytest.raises(ValidationError):
        CandidateResult.model_validate(_payload(metrics=metrics))


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="surprise"):
        CandidateResult.model_validate(_payload(surprise=1))


@pytest.mark.parametrize(
    "key",
    [
        "test_auc",
        "holdout_auc",
        "final_holdout_auc",
        "holdout-auc",
        "heldout_auc",
        "held_out_auc",
        "Hold Out AUC",
        "finalAuc",
        "prior_final_auc",
    ],
)
def test_metric_names_cannot_smuggle_holdout_scores(key: str) -> None:
    with pytest.raises(ValidationError, match=key):
        SearchMetrics(train={"auc": 0.7}, validation={"auc": 0.7, key: 0.9})


def test_objective_cannot_name_a_holdout() -> None:
    with pytest.raises(ValidationError):
        candidate_result(objective_metric="val_holdout_auc")


def test_a_hyperparameter_may_use_any_name() -> None:
    assert CandidateResult.model_validate(_payload(hyperparameters={"final_lr": 0.1}))


def test_test_size_is_a_legitimate_hyperparameter() -> None:
    assert CandidateResult.model_validate(_payload(hyperparameters={"test_size": 0.2}))


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_seed_and_count_do_not_coerce(value) -> None:
    with pytest.raises(ValidationError, match="random_seed"):
        CandidateResult.model_validate(_payload(random_seed=value))
    with pytest.raises(ValidationError, match="bootstrap_count"):
        CandidateResult.model_validate(_payload(bootstrap_count=value))


def test_nan_hyperparameter_is_rejected() -> None:
    with pytest.raises(ValidationError, match="finite"):
        CandidateResult.model_validate(_payload(hyperparameters={"a": {"b": [float("nan")]}}))
