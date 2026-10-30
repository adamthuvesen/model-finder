"""Rigour is a default that a person can override. An override is recorded and labelled."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agentic_ml_loop.candidate_result import (
    CandidateResult,
    CandidateRunner,
    FittedCandidate,
    SearchMetrics,
)
from agentic_ml_loop.cli.main import main
from agentic_ml_loop.cli.runner import RunnerConfig, run_runner_main
from agentic_ml_loop.ledger import read_ledger
from agentic_ml_loop.runner import ExperimentSeed
from agentic_ml_loop.splits import SearchSplits
from tests.conftest import SPEC, rows


def _quick(candidate_id: str, score: float) -> CandidateRunner:
    def runner(_splits: SearchSplits) -> FittedCandidate:
        return FittedCandidate(
            result=CandidateResult(
                candidate_id=candidate_id,
                objective_metric="val_auc",
                objective_score=score,
                metrics=SearchMetrics(validation={"auc": score}),
            ),
            score_rows=lambda _rows: {"auc": score},
        )

    return runner


@pytest.fixture
def config(code_root: Path) -> RunnerConfig:
    return RunnerConfig(
        seed=ExperimentSeed("experiment"),
        candidate_runners={"quick": _quick("quick", 0.6), "other": _quick("other", 0.7)},
        rows_loader=lambda seed: rows(seed=seed),
        rows_source=None,
        time_column="when",
        target_column="label",
        code_root=code_root,
    )


def _spec(experiment: Path) -> None:
    (experiment / "experiment.md").write_text(SPEC)


def _start(config: RunnerConfig, experiment: Path, *flags: str) -> None:
    assert run_runner_main(config, ["init-demo", "--experiment", str(experiment), *flags]) == 0
    _spec(experiment)


def _step(config: RunnerConfig, command: str, experiment: Path, *flags: str) -> int:
    return run_runner_main(config, [command, "--experiment", str(experiment), *flags])


def _complete(experiment: Path, selected: str = "quick") -> None:
    arguments = ["record-completion", "--experiment", str(experiment)]
    claim = ["--outcome", "supported", "--conclusion", "It helps.", "--selected", selected]
    assert main([*arguments, *claim, "--baseline", "quick"]) == 0


def test_an_experiment_runs_and_completes_with_no_holdout(
    config: RunnerConfig, tmp_path: Path, capsys
) -> None:
    experiment = tmp_path / "e"
    _start(config, experiment, "--no-holdout")
    assert not (experiment / "holdout_seal.json").exists()
    assert _step(config, "run-candidate", experiment, "--candidate", "quick") == 0
    assert _step(config, "run-candidate", experiment, "--candidate", "other") == 0
    assert main(["validate", str(experiment)]) == 0
    assert "No holdout is sealed" in capsys.readouterr().out

    assert main(["leaderboard", str(experiment)]) == 0
    board = capsys.readouterr().out
    assert board.index("other") < board.index("quick")

    _complete(experiment, "other")
    capsys.readouterr()
    assert main(["complete-check", str(experiment)]) == 0
    out = capsys.readouterr().out
    assert "Result: complete, exploratory" in out
    assert "No holdout was sealed" in out
    assert main(["complete-check", str(experiment), "--strict"]) == 1


def test_a_holdout_can_be_added_later_with_one_command(
    config: RunnerConfig, tmp_path: Path
) -> None:
    experiment = tmp_path / "e"
    _start(config, experiment, "--no-holdout")
    new_rows = tmp_path / "new.parquet"
    rows(count=12, seed=99, start="2026-06-01").to_parquet(new_rows, index=False)
    arguments = ["reseal", "--experiment", str(experiment), "--rows", str(new_rows)]
    assert main([*arguments, "--note", "Fresh rows."]) == 0
    assert _step(config, "run-candidate", experiment, "--candidate", "quick") == 0
    assert _step(config, "freeze-candidate", experiment, "--candidate", "quick") == 0
    assert _step(config, "evaluate-final", experiment) == 0
    _complete(experiment)
    assert main(["complete-check", str(experiment)]) == 0


def test_a_second_read_needs_a_reason_and_is_labelled(
    config: RunnerConfig, tmp_path: Path, capsys
) -> None:
    experiment = tmp_path / "e"
    _start(config, experiment)
    assert _step(config, "run-candidate", experiment, "--candidate", "quick") == 0
    assert _step(config, "freeze-candidate", experiment, "--candidate", "quick") == 0
    assert _step(config, "evaluate-final", experiment) == 0
    assert _step(config, "evaluate-final", experiment, "--allow-reread") == 1
    assert "needs --reason" in capsys.readouterr().err

    flags = ["--allow-reread", "--reason", "Fixed a scoring bug."]
    assert _step(config, "evaluate-final", experiment, *flags) == 0
    notes = [entry.provenance_note for entry in read_ledger() if entry.event == "evaluated"]
    assert notes == [None, "Fixed a scoring bug."]
    assert (experiment / "work/rereads/0001/final_evaluation.json").is_file()

    _complete(experiment)
    capsys.readouterr()
    assert main(["complete-check", str(experiment)]) == 0
    out = capsys.readouterr().out
    assert "Result: complete, exploratory" in out
    assert "Fixed a scoring bug." in out


def test_holdout_based_selection_is_allowed_only_when_labelled(
    config: RunnerConfig, tmp_path: Path, capsys
) -> None:
    experiment = tmp_path / "e"
    _start(config, experiment)
    for candidate in ("quick", "other"):
        assert _step(config, "run-candidate", experiment, "--candidate", candidate) == 0
    assert _step(config, "freeze-candidate", experiment, "--candidate", "quick") == 0
    assert _step(config, "evaluate-final", experiment) == 0

    assert _step(config, "freeze-candidate", experiment, "--candidate", "other") == 1
    assert "holdout-informed" in capsys.readouterr().err
    flags = ["--allow-reread", "--reason", "Comparing on the holdout."]
    assert _step(config, "freeze-candidate", experiment, "--candidate", "other", *flags) == 0
    assert _step(config, "evaluate-final", experiment) == 1
    assert "allow-reread" in capsys.readouterr().err
    assert _step(config, "evaluate-final", experiment, *flags) == 0

    _complete(experiment, "other")
    capsys.readouterr()
    assert main(["complete-check", str(experiment)]) == 0
    out = capsys.readouterr().out
    assert "Result: complete, exploratory" in out
    assert "Search continued after reading the holdout: Comparing on the holdout." in out


def test_editing_a_helper_warns_and_a_strict_check_fails(
    config: RunnerConfig, tmp_path: Path, capsys
) -> None:
    experiment = tmp_path / "e"
    _start(config, experiment)
    assert _step(config, "run-candidate", experiment, "--candidate", "quick") == 0
    assert _step(config, "run-candidate", experiment, "--candidate", "other") == 0
    assert _step(config, "freeze-candidate", experiment, "--candidate", "other") == 0
    (config.code_root / "helper.py").write_text("VALUE = 2\n")
    assert _step(config, "evaluate-final", experiment) == 0
    assert "Code or dependencies changed" in capsys.readouterr().err

    _complete(experiment, "other")
    capsys.readouterr()
    assert main(["complete-check", str(experiment)]) == 0
    out = capsys.readouterr().out
    assert "Result: complete\n" in out
    assert "Code or dependencies changed" in out
    assert main(["complete-check", str(experiment), "--strict"]) == 1


def test_a_changed_rerun_still_blocks_before_the_read(
    config: RunnerConfig, tmp_path: Path, capsys
) -> None:
    experiment = tmp_path / "e"
    _start(config, experiment)
    assert _step(config, "run-candidate", experiment, "--candidate", "quick") == 0
    assert _step(config, "freeze-candidate", experiment, "--candidate", "quick") == 0
    changed = replace(config, candidate_runners={"quick": _quick("quick", 0.9)})
    assert _step(changed, "evaluate-final", experiment) == 1
    assert "changed frozen selection field" in capsys.readouterr().err
    assert read_ledger() == []


def _inconclusive(experiment: Path) -> None:
    arguments = ["record-completion", "--experiment", str(experiment)]
    assert main([*arguments, "--outcome", "inconclusive", "--conclusion", "Unclear."]) == 0


def _board(experiment: Path, capsys) -> str:
    capsys.readouterr()
    assert main(["leaderboard", str(experiment)]) == 0
    return capsys.readouterr().out


def test_an_inconclusive_claim_with_no_holdout_is_labelled(
    config: RunnerConfig, tmp_path: Path, capsys
) -> None:
    experiment = tmp_path / "e"
    _start(config, experiment, "--no-holdout")
    assert _step(config, "run-candidate", experiment, "--candidate", "quick") == 0
    _inconclusive(experiment)
    capsys.readouterr()
    assert main(["complete-check", str(experiment)]) == 0
    assert "Result: complete, exploratory" in capsys.readouterr().out


def test_an_inconclusive_claim_after_search_reopened_is_labelled(
    config: RunnerConfig, tmp_path: Path, capsys
) -> None:
    experiment = tmp_path / "e"
    _start(config, experiment)
    assert _step(config, "run-candidate", experiment, "--candidate", "quick") == 0
    assert _step(config, "freeze-candidate", experiment, "--candidate", "quick") == 0
    assert _step(config, "evaluate-final", experiment) == 0
    flags = ["--allow-reread", "--reason", "Trying another."]
    assert _step(config, "run-candidate", experiment, "--candidate", "other", *flags) == 0
    _inconclusive(experiment)
    capsys.readouterr()
    assert main(["complete-check", str(experiment)]) == 0
    assert "Result: complete, exploratory" in capsys.readouterr().out


def test_leaderboard_labels_a_run_with_no_holdout(
    config: RunnerConfig, tmp_path: Path, capsys
) -> None:
    experiment = tmp_path / "e"
    _start(config, experiment, "--no-holdout")
    assert _step(config, "run-candidate", experiment, "--candidate", "quick") == 0
    assert "Exploratory: no holdout is sealed." in _board(experiment, capsys)


def test_leaderboard_labels_a_reread(config: RunnerConfig, tmp_path: Path, capsys) -> None:
    experiment = tmp_path / "e"
    _start(config, experiment)
    assert _step(config, "run-candidate", experiment, "--candidate", "quick") == 0
    assert "Exploratory" not in _board(experiment, capsys)
    assert _step(config, "freeze-candidate", experiment, "--candidate", "quick") == 0
    assert _step(config, "evaluate-final", experiment) == 0
    assert "Exploratory" not in _board(experiment, capsys)
    flags = ["--allow-reread", "--reason", "Fixed a scoring bug."]
    assert _step(config, "evaluate-final", experiment, *flags) == 0
    assert "Exploratory: a holdout read" in _board(experiment, capsys)
