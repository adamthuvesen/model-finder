"""Every override is recorded and labelled, on every surface that shows a result."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from agentic_ml_loop.candidate_result import CandidateResult, SearchMetrics
from agentic_ml_loop.cli.main import main
from agentic_ml_loop.cli.runner import RunnerConfig, run_runner_main
from agentic_ml_loop.runner import ExperimentSeed
from tests.conftest import SPEC, fitted, rows
from tests.test_usable_defaults import _quick


@pytest.fixture
def config(code_root: Path) -> RunnerConfig:
    return RunnerConfig(
        seed=ExperimentSeed("experiment"),
        candidate_runners={"quick": _quick("quick", 0.5), "full": fitted},
        rows_loader=lambda seed: rows(seed=seed),
        time_column="when",
        target_column="label",
        code_root=code_root,
    )


def _step(config: RunnerConfig, command: str, experiment: Path, *flags: str) -> int:
    return run_runner_main(config, [command, "--experiment", str(experiment), *flags])


def _started(config: RunnerConfig, experiment: Path, *flags: str) -> Path:
    assert _step(config, "init-demo", experiment, *flags) == 0
    (experiment / "experiment.md").write_text(SPEC)
    assert _step(config, "run-candidate", experiment, "--candidate", "quick") == 0
    return experiment


def _replay(experiment: Path, tmp_path: Path) -> str:
    output = tmp_path / "replay.html"
    assert main(["replay", str(experiment), "--output", str(output)]) == 0
    return output.read_text()


def test_replay_shows_the_label_with_no_holdout(config: RunnerConfig, tmp_path: Path) -> None:
    experiment = _started(config, tmp_path / "e", "--no-holdout")
    assert 'id="exploratory-label"' in _replay(experiment, tmp_path)


def test_replay_shows_the_label_after_a_reread(config: RunnerConfig, tmp_path: Path) -> None:
    experiment = _started(config, tmp_path / "e")
    assert 'id="exploratory-label"' not in _replay(experiment, tmp_path)
    assert _step(config, "freeze-candidate", experiment, "--candidate", "quick") == 0
    assert _step(config, "evaluate-final", experiment) == 0
    flags = ["--allow-reread", "--reason", "A scoring bug."]
    assert _step(config, "evaluate-final", experiment, *flags) == 0
    assert 'id="exploratory-label"' in _replay(experiment, tmp_path)


def _edit_run_record(experiment: Path, score: float) -> None:
    record_path = next((experiment / "work/runs").glob("*.json"))
    record = json.loads(record_path.read_text())
    record["result"]["objective_score"] = score
    record["result"]["metrics"]["validation"]["auc"] = score
    record_path.write_text(json.dumps(record))


def test_rebuild_refuses_an_edited_run_record_and_names_the_change(
    config: RunnerConfig, tmp_path: Path, capsys
) -> None:
    experiment = _started(config, tmp_path / "e")
    before = (experiment / "results.json").read_text()
    _edit_run_record(experiment, 0.99)
    capsys.readouterr()
    assert main(["rebuild-results", str(experiment)]) == 1
    err = capsys.readouterr().err
    assert "objective_score: 0.5 -> 0.99" in err
    assert (experiment / "results.json").read_text() == before


def test_a_rebuild_with_changes_is_recorded_and_labelled(
    config: RunnerConfig, tmp_path: Path, capsys
) -> None:
    experiment = _started(config, tmp_path / "e", "--no-holdout")
    _edit_run_record(experiment, 0.99)
    flags = ["--allow-changes", "--reason", "The record is right."]
    assert main(["rebuild-results", str(experiment), *flags]) == 0
    arguments = ["record-completion", "--experiment", str(experiment)]
    assert main([*arguments, "--outcome", "inconclusive", "--conclusion", "Unclear."]) == 0
    capsys.readouterr()
    assert main(["complete-check", str(experiment)]) == 0
    assert "A rebuild changed recorded results: The record is right." in capsys.readouterr().out


def _pull(experiment: Path, database: Path, *flags: str) -> int:
    query = "SELECT * FROM events LIMIT 40"
    return main(
        [
            "pull",
            "--experiment",
            str(experiment),
            "--source",
            "duckdb",
            "--database",
            str(database),
            "--query",
            query,
            *flags,
        ]
    )


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "events.duckdb"
    connection = duckdb.connect(str(path))
    frame = rows(count=60).rename(columns={"when": "when_at"})
    connection.register("frame", frame)
    connection.execute("CREATE TABLE events AS SELECT * FROM frame")
    connection.close()
    return path


def test_a_nondeterministic_pull_needs_a_reason_and_is_labelled(
    tmp_path: Path, database: Path, capsys
) -> None:
    experiment = tmp_path / "e"
    assert _pull(experiment, database) == 1
    assert _pull(experiment, database, "--allow-nondeterministic") == 1
    assert "needs --reason" in capsys.readouterr().err
    assert _pull(experiment, database, "--allow-nondeterministic", "--reason", "A sample.") == 0
    manifest = json.loads((experiment / "data/dataset_manifest.json").read_text())
    assert manifest["nondeterministic_reason"] == "A sample."

    config = RunnerConfig(
        seed=ExperimentSeed("experiment"),
        candidate_runners={"quick": _quick("quick", 0.5)},
        rows_loader=lambda seed: pd.DataFrame(),
        time_column="when_at",
        target_column="label",
        code_root=tmp_path,
    )
    seal = ["seal", "--experiment", str(experiment), "--time-column", "when_at"]
    assert main(seal) == 0
    assert _step(config, "init-demo", experiment) == 0
    (experiment / "experiment.md").write_text(SPEC)
    assert _step(config, "run-candidate", experiment, "--candidate", "quick") == 0
    capsys.readouterr()
    assert main(["leaderboard", str(experiment)]) == 0
    assert "Exploratory" in capsys.readouterr().out
    arguments = ["record-completion", "--experiment", str(experiment)]
    claim = ["--outcome", "inconclusive", "--conclusion", "Unclear."]
    assert main([*arguments, *claim]) == 0
    capsys.readouterr()
    assert main(["complete-check", str(experiment)]) == 0
    assert "The data pull is not reproducible: A sample." in capsys.readouterr().out


def test_reason_without_its_flag_is_an_error(config: RunnerConfig, tmp_path: Path, capsys) -> None:
    experiment = _started(config, tmp_path / "e")
    run = ["--candidate", "quick", "--reason", "Dropped."]
    assert _step(config, "run-candidate", experiment, *run) == 1
    assert "--reason does nothing without --allow-reread" in capsys.readouterr().err


def test_final_after_a_reseal_lists_its_exploratory_reason(
    config: RunnerConfig, tmp_path: Path, capsys
) -> None:
    experiment = _started(config, tmp_path / "e")
    new_rows = tmp_path / "new.parquet"
    rows(count=12, seed=99, start="2026-06-01").to_parquet(new_rows, index=False)
    arguments = ["reseal", "--experiment", str(experiment), "--rows", str(new_rows)]
    assert main([*arguments, "--note", "New rows."]) == 0
    assert _step(config, "freeze-candidate", experiment, "--candidate", "quick") == 0
    capsys.readouterr()
    assert _step(config, "evaluate-final", experiment) == 0
    final = json.loads(capsys.readouterr().out)
    assert any("1 earlier holdouts" in reason for reason in final["exploratory_reasons"])


def test_a_hyperparameter_naming_a_holdout_score_warns(
    config: RunnerConfig, tmp_path: Path, capsys
) -> None:
    from agentic_ml_loop.validation import hyperparameter_warnings

    result = CandidateResult(
        candidate_id="c",
        objective_metric="val_auc",
        objective_score=0.5,
        metrics=SearchMetrics(validation={"auc": 0.5}),
        hyperparameters={"holdout_auc": 0.91, "nested": {"test_auc": 0.9}, "test_size": 0.2},
    )
    assert len(hyperparameter_warnings([result])) == 2
    assert "holdout_auc" in hyperparameter_warnings([result])[0]


def test_validate_after_a_pull_names_the_next_step(tmp_path: Path, database: Path, capsys) -> None:
    experiment = tmp_path / "e"
    flags = ["--allow-nondeterministic", "--reason", "A sample."]
    assert _pull(experiment, database, *flags) == 0
    capsys.readouterr()
    assert main(["validate", str(experiment)]) == 1
    err = capsys.readouterr().err
    assert "Next: run `experiment seal" in err
    assert "init-demo" in err


def test_rebuild_takes_the_experiment_as_a_positional_argument(
    config: RunnerConfig, tmp_path: Path
) -> None:
    experiment = _started(config, tmp_path / "e")
    assert main(["rebuild-results", str(experiment)]) == 0


def test_rebuild_restores_a_missing_leaderboard_without_a_flag(
    config: RunnerConfig, tmp_path: Path
) -> None:
    experiment = _started(config, tmp_path / "e")
    results = experiment / "results.json"
    recorded = results.read_text()
    results.unlink()
    assert main(["rebuild-results", str(experiment)]) == 0
    assert results.read_text() == recorded


def test_readme_csv_recipe_reaches_a_leaderboard(tmp_path: Path, monkeypatch, capsys) -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    recipe = readme.split("## Use your own CSV")[1].split("```python\n")[1].split("```")[0]
    package = tmp_path / "my_model"
    package.mkdir()
    (package / "run.py").write_text(recipe)
    monkeypatch.chdir(tmp_path)
    rows(count=60).assign(x=lambda frame: frame["signal"]).to_csv("rows.csv", index=False)
    run = [sys.executable, "my_model/run.py"]
    experiment = ["--experiment", "experiments/my_run"]
    for arguments in (
        ["init-demo", *experiment, "--no-holdout"],
        ["run-candidate", *experiment, "--candidate", "logreg"],
    ):
        subprocess.run([*run, *arguments], check=True, capture_output=True)
    assert main(["leaderboard", "experiments/my_run"]) == 0
    out = capsys.readouterr().out
    assert "logreg" in out
    assert "Exploratory: no holdout" in out
