"""`experiment new` gives a working experiment, and `experiment next` always has a suggestion."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_finder.cli.main import main
from model_finder.cli.next_step import next_steps
from model_finder.cli.runner import run_runner_main
from model_finder.cli.scaffold import load_runner

ROOT = Path(__file__).parents[2]


def _csv(path: Path, *, binary: bool = True, columns: tuple[str, str] = ("date", "label")) -> Path:
    rng = np.random.default_rng(1)
    signal = rng.normal(size=200)
    target = (signal + rng.normal(size=200) > 0).astype(int) if binary else signal * 3.0
    pd.DataFrame(
        {
            columns[0]: pd.date_range("2025-01-01", periods=200).strftime("%Y-%m-%d"),
            "signal": signal,
            columns[1]: target,
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def consumer(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    _csv(tmp_path / "rows.csv")
    return tmp_path


def test_one_command_reaches_a_leaderboard(consumer: Path, capsys) -> None:
    assert main(["new", "churn", "--csv", "rows.csv"]) == 0
    out = capsys.readouterr().out
    assert out.index("linear") < out.index("baseline")
    assert "Target: label. Split: date." in out
    assert "Next: Try another candidate." in out
    assert (consumer / "experiments/churn/holdout_seal.json").is_file()
    assert "CHANGE" in (consumer / "experiments/churn/model/candidates.py").read_text()


def test_a_numeric_target_scaffolds_regression(consumer: Path, capsys) -> None:
    _csv(consumer / "revenue.csv", binary=False, columns=("when", "revenue"))
    arguments = ["new", "rev", "--csv", "revenue.csv", "--target", "revenue", "--no-holdout"]
    assert main(arguments) == 0
    out = capsys.readouterr().out
    assert "val_rmse" in out
    assert "Exploratory: no holdout" in out
    assert main(["validate", "experiments/rev"]) == 0


def test_a_file_with_no_time_column_keeps_row_order(consumer: Path, capsys) -> None:
    rows = pd.read_csv("rows.csv").assign(region="north").drop(columns="date")
    rows.to_csv("plain.csv", index=False)
    assert main(["new", "plain", "--csv", "plain.csv"]) == 0
    assert "Split: file row order." in capsys.readouterr().out


def test_the_generated_runner_works_as_a_script(consumer: Path) -> None:
    assert main(["new", "churn", "--csv", "rows.csv"]) == 0
    runner = [sys.executable, "experiments/churn/model/run.py"]
    done = subprocess.run(
        [*runner, "run-candidate", "--candidate", "linear"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(done.stdout)["candidate_id"] == "linear"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["new", "x", "--csv", "absent.csv"], "absent.csv: no such file. Fix:"),
        (
            ["new", "x", "--csv", "rows.csv", "--target", "nope"],
            "--target nope is not a column. Fix:",
        ),
        (["new", "x", "--csv", "rows.csv", "--target", "date"], "is not numeric. Fix:"),
        (
            ["new", "x", "--csv", "rows.csv", "--time-column", "nope"],
            "--time-column nope is not a column. Fix:",
        ),
        (["candidate", "new", "forest"], "found 0 candidates.py files. Fix:"),
    ],
)
def test_scaffold_mistakes_name_a_fix(consumer: Path, arguments, expected: str, capsys) -> None:
    assert main(arguments) == 1
    assert expected in capsys.readouterr().err
    assert not (consumer / "experiments/x").exists()


def test_new_refuses_an_existing_experiment(consumer: Path, capsys) -> None:
    assert main(["new", "churn", "--csv", "rows.csv"]) == 0
    assert main(["new", "churn", "--csv", "rows.csv"]) == 1
    assert "already exists. Fix:" in capsys.readouterr().err


def test_a_new_candidate_is_one_command_and_runs(consumer: Path, capsys) -> None:
    assert main(["new", "churn", "--csv", "rows.csv"]) == 0
    assert main(["candidate", "new", "wide-model"]) == 0
    assert "Added wide-model" in capsys.readouterr().out
    config = load_runner(consumer / "experiments/churn/model")
    assert run_runner_main(config, ["run-candidate", "--candidate", "wide-model"]) == 0
    assert main(["candidate", "new", "wide-model"]) == 1
    assert "already exists. Fix:" in capsys.readouterr().err


def _steps_text(experiment: Path) -> str:
    return " ".join(f"{step.text} {step.command}" for step in next_steps(experiment))


def test_next_follows_the_state_and_offers_alternatives(consumer: Path, capsys) -> None:
    experiment = consumer / "experiments/churn"
    assert "experiment new" in _steps_text(experiment)
    assert main(["new", "churn", "--csv", "rows.csv"]) == 0
    capsys.readouterr()
    text = _steps_text(experiment)
    assert "freeze-candidate --candidate linear" in text
    assert "experiment candidate new" in text

    config = load_runner(experiment / "model")
    assert run_runner_main(config, ["freeze-candidate", "--candidate", "linear"]) == 0
    text = _steps_text(experiment)
    assert "evaluate-final" in text
    assert "experiment candidate new" in text

    assert run_runner_main(config, ["evaluate-final"]) == 0
    assert "record-completion" in _steps_text(experiment)

    arguments = ["record-completion", "--experiment", str(experiment), "--outcome", "supported"]
    assert main([*arguments, "--conclusion", "It helps."]) == 0
    assert "complete-check" in _steps_text(experiment)

    capsys.readouterr()
    assert main(["next"]) == 0
    assert capsys.readouterr().out.startswith("Next: This work reads:")


def test_next_needs_no_experiment_and_never_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["next"]) == 0
    assert "experiment new my_run --csv rows.csv" in capsys.readouterr().out
