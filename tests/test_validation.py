from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_ml_loop.io import write_json
from agentic_ml_loop.results import write_results
from agentic_ml_loop.runner import ExperimentSeed, init_experiment_dir
from agentic_ml_loop.spec import get_objective_direction, get_objective_metric
from agentic_ml_loop.validation import validate_experiment
from tests.conftest import SPEC, recorded_result


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    path = init_experiment_dir(tmp_path / "experiment", ExperimentSeed("experiment"))
    (path / "experiment.md").write_text(SPEC)
    return path


def _errors(path: Path, **options: bool) -> list[str]:
    return [m for m in validate_experiment(path, **options) if not m.startswith("warning:")]


def test_seeded_experiment_is_valid(experiment: Path) -> None:
    assert validate_experiment(experiment) == []


def test_missing_directory_and_files_are_reported(experiment: Path, tmp_path: Path) -> None:
    assert "does not exist" in validate_experiment(tmp_path / "absent")[0]
    (experiment / "experiment.md").unlink()
    assert "Missing required file" in validate_experiment(experiment)[0]


def test_invalid_results_are_errors(experiment: Path) -> None:
    (experiment / "results.json").write_text("{not json")
    assert _errors(experiment)
    (experiment / "results.json").write_text('{"a": 1}')
    assert _errors(experiment)


def test_committed_shape_with_test_metrics_is_rejected(experiment: Path) -> None:
    payload = recorded_result().model_dump()
    payload["metrics"]["test"] = {"auc": 0.9}
    write_json(experiment / "results.json", [payload])
    assert any("test" in message for message in _errors(experiment))


def test_strict_completion_requires_a_result(experiment: Path) -> None:
    assert _errors(experiment, strict_completion=True)


def test_objective_mismatch_is_an_error(experiment: Path) -> None:
    write_results(experiment, [recorded_result(objective_metric="val_r2")])
    assert any("declares val_auc" in message for message in _errors(experiment))


def test_direction_mismatch_is_an_error(experiment: Path) -> None:
    (experiment / "experiment.md").write_text(SPEC + "\nobjective_direction: minimize\n")
    write_results(experiment, [recorded_result()])
    assert any("declares minimize" in message for message in _errors(experiment))


def test_large_gap_warns(experiment: Path) -> None:
    result = recorded_result()
    gapped = result.model_copy(
        update={
            "metrics": result.metrics.model_copy(update={"train": {"auc": 0.9}}),
            "train_validation_gap": 0.2,
        }
    )
    write_results(experiment, [gapped])
    assert any("train-validation gap" in message for message in validate_experiment(experiment))


def test_stray_files_warn(experiment: Path) -> None:
    (experiment / "scores.csv").write_text("x")
    assert "stray files" in validate_experiment(experiment)[0]


def test_duplicate_ids_are_rejected(experiment: Path) -> None:
    (experiment / "results.json").write_text(
        json.dumps([recorded_result().model_dump(), recorded_result().model_dump()])
    )
    assert any("duplicate candidate_id" in message for message in _errors(experiment))


def test_objective_metric_parsing(tmp_path: Path) -> None:
    assert get_objective_metric(tmp_path) is None
    spec = tmp_path / "experiment.md"
    spec.write_text("## Objective Metric\n\nNo token here.\n")
    assert get_objective_metric(tmp_path) is None
    spec.write_text("## Objective Metric\n\nValidation R2 (val_r2).\n")
    assert get_objective_metric(tmp_path) == "val_r2"


def test_objective_direction_parsing(tmp_path: Path) -> None:
    assert get_objective_direction(tmp_path) == "maximize"
    spec = tmp_path / "experiment.md"
    spec.write_text("objective_direction: minimize\n")
    assert get_objective_direction(tmp_path) == "minimize"
    spec.write_text("objective_direction: minimize\nobjective_direction: maximize\n")
    with pytest.raises(ValueError, match="Conflicting"):
        get_objective_direction(tmp_path)
    spec.write_text("objective_direction: sideways\n")
    with pytest.raises(ValueError, match="minimize or maximize"):
        get_objective_direction(tmp_path)
