from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from agentic_ml_loop.candidate_result import (
    CandidateResult,
    FittedCandidate,
    SearchMetrics,
    metric_key,
    train_validation_gap,
)
from agentic_ml_loop.holdout import seal_holdout
from agentic_ml_loop.results import ComparisonContext, RecordedResult
from agentic_ml_loop.runner import ExperimentSeed, init_experiment_dir, save_candidate_result
from agentic_ml_loop.splits import (
    FrameIdentity,
    SearchDataIdentity,
    SearchSplits,
    temporal_split,
)

_IDENTITY = FrameIdentity(sha256="a" * 64, rows=1, columns=1)
SPEC = "# Experiment\n\n## Objective Metric\n\nValidation AUC (`val_auc`).\n"


@pytest.fixture(autouse=True)
def isolated_holdout_store(tmp_path_factory, monkeypatch) -> Path:
    """No test may touch the real holdout store."""
    root = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_DATA_HOME", str(root))
    return root


def candidate_result(
    candidate_id: str = "candidate-a",
    objective_metric: str = "val_auc",
    score: float = 0.7,
    validation: dict[str, float] | None = None,
) -> CandidateResult:
    validation = validation or {metric_key(objective_metric): score}
    metrics = SearchMetrics(train=dict(validation), validation=validation)
    return CandidateResult(
        candidate_id=candidate_id,
        model_family="constant",
        feature_set="none",
        objective_metric=objective_metric,
        objective_score=score,
        split_strategy="time",
        notes="",
        metrics=metrics,
        hyperparameters={},
        selected_features=[],
        random_seed=None,
        bootstrap_count=None,
        train_validation_gap=train_validation_gap(metrics, objective_metric),
    )


def recorded_result(
    candidate_id: str = "candidate-a",
    objective_metric: str = "val_auc",
    score: float = 0.7,
    validation: dict[str, float] | None = None,
) -> RecordedResult:
    result = candidate_result(candidate_id, objective_metric, score, validation)
    run_id = str(uuid4())
    return RecordedResult(
        **result.model_dump(),
        comparison_context=ComparisonContext(
            objective_metric=objective_metric,
            objective_direction="maximize",
            spec_sha256="b" * 64,
            data=SearchDataIdentity(train=_IDENTITY, validation=_IDENTITY),
        ),
        run_id=run_id,
        run_evidence=f"work/runs/{run_id}.json",
    )


def rows(count: int = 60, seed: int = 8, start: str = "2025-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "when": pd.date_range(start, periods=count),
            "signal": rng.normal(size=count),
            "label": np.resize([0, 1], count),
        }
    )


def fitted(splits: SearchSplits, candidate_id: str = "candidate-a") -> FittedCandidate:
    """A deterministic candidate whose score depends on its rows."""
    score = float(splits.validation["signal"].abs().mean())
    train_score = float(splits.train["signal"].abs().mean())
    metrics = SearchMetrics(train={"auc": train_score}, validation={"auc": score})
    return FittedCandidate(
        result=CandidateResult(
            candidate_id=candidate_id,
            model_family="constant",
            feature_set="none",
            objective_metric="val_auc",
            objective_score=score,
            split_strategy="time",
            notes="",
            metrics=metrics,
            hyperparameters={},
            selected_features=["signal"],
            random_seed=None,
            bootstrap_count=None,
            train_validation_gap=train_validation_gap(metrics, "val_auc"),
        ),
        score_rows=lambda frame: {"auc": float(frame["signal"].abs().mean())},
    )


@pytest.fixture
def code_root(tmp_path: Path) -> Path:
    root = tmp_path / "candidate_code"
    root.mkdir()
    (root / "candidate_module.py").write_text("VERSION = 1\n")
    return root


@pytest.fixture
def sealed(tmp_path: Path) -> Path:
    """An initialized experiment with a sealed holdout and no results."""
    experiment = tmp_path / "experiment"
    init_experiment_dir(experiment, ExperimentSeed("experiment"))
    (experiment / "experiment.md").write_text(SPEC)
    search, holdout = temporal_split(rows(), "when")
    seal_holdout(experiment, search, holdout)
    return experiment


@pytest.fixture
def scored(sealed: Path, code_root: Path) -> Path:
    """A sealed experiment with one recorded candidate."""
    save_candidate_result(sealed, "candidate-a", {"candidate-a": fitted}, code_root)
    return sealed
