"""Each common mistake names its fix, in one format: `<file>:<line>: <what>. Fix: <change>`."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from model_finder.candidate_result import CandidateResult, SearchMetrics
from model_finder.cli.errors import describe
from model_finder.cli.main import main
from model_finder.final_evaluation import evaluate_final
from model_finder.ledger import read_ledger, store_root
from model_finder.results import load_results
from tests.conftest import fitted, result_fields

ROOT = Path(__file__).parents[1]


def _result(**fields: object) -> None:
    CandidateResult.model_validate({**result_fields(), **fields})


def _message(call, *arguments) -> str:
    with pytest.raises((ValueError, OSError)) as caught:
        call(*arguments)
    return describe(caught.value)


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (
            lambda: SearchMetrics(validation={"auc": 0.7, "holdout_auc": 0.9}),
            r"metric holdout_auc looks like a holdout score\. Fix: remove it, and use evaluate-final",
        ),
        (lambda: _result(objective_metric="auc"), r"not a validation metric\. Fix: rename it"),
        (lambda: _result(objective_score=0.1), r"differs from .*\. Fix: set objective_score"),
        (lambda: _result(metrics={"validation": {"r2": 0.1}}), r"has no auc.*\. Fix: add auc"),
        (lambda: _result(unknown=1), r"unknown is not a known field\. Fix: remove it"),
        (
            lambda: _result(objective_score=float("nan")),
            r"not a finite number\. Fix: check the predictions",
        ),
        (
            lambda: CandidateResult.model_validate({"candidate_id": "a"}),
            r"objective_metric is required\. Fix: set objective_metric",
        ),
        (lambda: _result(candidate_id=" "), r"candidate_id must not be blank\. Fix: pass text"),
        (
            lambda: _result(train_validation_gap=9.0, metrics={"validation": {"auc": 0.7}}),
            r"train_validation_gap does not match.*\. Fix: remove",
        ),
    ],
)
def test_a_wrong_candidate_result_names_its_fix(build, expected: str) -> None:
    with pytest.raises(ValidationError) as caught:
        build()
    assert re.search(expected, describe(caught.value))


def test_a_mistake_in_a_candidate_names_the_line_in_that_file() -> None:
    def candidate() -> None:
        SearchMetrics(validation={"auc": 0.7, "holdout_auc": 0.9})

    with pytest.raises(ValidationError) as caught:
        candidate()
    first_line = describe(caught.value).splitlines()[0]
    assert re.match(r"tests/test_feedback\.py:\d+: metric holdout_auc looks like", first_line)


def test_a_corrupt_leaderboard_names_the_repair(sealed: Path) -> None:
    (sealed / "results.json").write_text("{ damaged")
    message = _message(load_results, sealed)
    assert message.startswith("results.json: ")
    assert "experiment rebuild-results" in message


def test_a_corrupt_ledger_names_the_file_line_and_fix(scored: Path) -> None:
    ledger = store_root() / "ledger.jsonl"
    ledger.write_text("not json\n")
    message = _message(read_ledger)
    assert message.startswith(f"{ledger}:1: ")
    assert "Fix: " in message


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            ["validate", "absent"],
            "absent: experiment directory does not exist. Fix: run `experiment new",
        ),
        (["leaderboard", "absent"], "experiment directory does not exist. Fix:"),
        (["reseal", "--experiment", "absent", "--rows", "absent.parquet", "--note", "x"], "Fix:"),
        (
            [
                "record-completion",
                "--experiment",
                "absent",
                "--outcome",
                "inconclusive",
                "--conclusion",
                "x",
            ],
            "Fix:",
        ),
    ],
)
def test_commands_on_a_missing_experiment_name_a_fix(
    arguments: list[str], expected: str, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    main(arguments)
    captured = capsys.readouterr()
    assert expected in captured.out + captured.err


def test_final_before_a_freeze_names_the_command(scored: Path) -> None:
    message = _message(evaluate_final, scored, {"candidate-a": fitted})
    assert "No frozen selection. Fix: run `freeze-candidate --candidate <id>`" in message


def test_an_error_with_no_known_fix_gets_no_generic_one() -> None:
    assert describe(RuntimeError("boom")) == "boom"


def test_static_analysis_messages_name_a_fix() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    banned = config["tool"]["ruff"]["lint"]["flake8-tidy-imports"]["banned-api"]
    assert all("Fix:" in rule["msg"] for rule in banned.values())
    contracts = config["tool"]["importlinter"]["contracts"]
    assert all("Fix:" in contract["name"] for contract in contracts)
    assert all("Fix:" in rule.read_text() for rule in (ROOT / "rules").glob("*.yml"))
