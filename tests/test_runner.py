from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from model_finder.final_evaluation import evaluate_final, freeze_candidate
from model_finder.results import load_results
from model_finder.run_evidence import RunEvidence, run_evidence_staleness
from model_finder.runner import ExperimentSeed, init_experiment_dir, save_candidate_result
from model_finder.splits import SearchSplits
from tests.conftest import fitted


def test_missing_experiment_is_reported(tmp_path: Path, code_root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="experiment directory"):
        save_candidate_result(tmp_path / "missing", "a", {"a": fitted}, code_root)


def test_unknown_candidate_is_reported_before_loading_data(tmp_path: Path, code_root: Path) -> None:
    with pytest.raises(ValueError, match="unknown candidate b"):
        save_candidate_result(tmp_path, "b", {"a": fitted}, code_root)


def test_unsealed_experiment_has_no_search_rows(tmp_path: Path, code_root: Path) -> None:
    experiment = init_experiment_dir(tmp_path / "fresh", ExperimentSeed("fresh"))
    with pytest.raises(FileNotFoundError, match="No search rows"):
        save_candidate_result(experiment, "candidate-a", {"candidate-a": fitted}, code_root)


def test_run_records_evidence_and_ranks_results(sealed: Path, code_root: Path) -> None:
    def weak(splits: SearchSplits):
        result = fitted(splits, "weak")
        return replace(
            result,
            result=result.result.model_copy(
                update={
                    "objective_score": 0.01,
                    "metrics": result.result.metrics.model_copy(
                        update={"validation": {"auc": 0.01}}
                    ),
                    "train_validation_gap": result.result.metrics.train["auc"] - 0.01,
                }
            ),
        )

    runners = {"candidate-a": fitted, "weak": weak}
    save_candidate_result(sealed, "weak", runners, code_root)
    recorded = save_candidate_result(sealed, "candidate-a", runners, code_root)

    assert [result.candidate_id for result in load_results(sealed)] == ["candidate-a", "weak"]
    evidence = RunEvidence.model_validate_json((sealed / recorded.run_evidence).read_text())
    assert evidence.result == recorded
    assert not Path(evidence.code_fingerprint.code_root).is_absolute()


def test_rerun_replaces_the_entry_and_keeps_old_evidence(scored: Path, code_root: Path) -> None:
    first = load_results(scored)[0]
    second = save_candidate_result(scored, "candidate-a", {"candidate-a": fitted}, code_root)
    assert [result.run_id for result in load_results(scored)] == [second.run_id]
    assert (scored / first.run_evidence).is_file()


def test_candidate_id_must_match_the_request(sealed: Path, code_root: Path) -> None:
    runners = {"candidate-a": lambda splits: fitted(splits, "other")}
    with pytest.raises(ValueError, match="returned id"):
        save_candidate_result(sealed, "candidate-a", runners, code_root)


def test_objective_must_match_the_specification(sealed: Path, code_root: Path) -> None:
    (sealed / "experiment.md").write_text("## Objective Metric\n\n(val_r2)\n")
    with pytest.raises(ValueError, match="the objective is"):
        save_candidate_result(sealed, "candidate-a", {"candidate-a": fitted}, code_root)


def test_direction_must_match_the_specification(sealed: Path, code_root: Path) -> None:
    (sealed / "experiment.md").write_text("objective_direction: minimize\n")
    with pytest.raises(ValueError, match="the direction is"):
        save_candidate_result(sealed, "candidate-a", {"candidate-a": fitted}, code_root)


def test_candidate_that_mutates_search_rows_is_rejected(sealed: Path, code_root: Path) -> None:
    def mutating(splits: SearchSplits):
        splits.train.loc[0, "signal"] = 999.0
        return fitted(splits)

    with pytest.raises(ValueError, match="changed the search rows"):
        save_candidate_result(sealed, "candidate-a", {"candidate-a": mutating}, code_root)


def test_changed_specification_cannot_join_the_leaderboard(scored: Path, code_root: Path) -> None:
    (scored / "experiment.md").write_text((scored / "experiment.md").read_text() + "\nmore\n")
    runners = {"candidate-a": fitted, "other": lambda splits: fitted(splits, "other")}
    with pytest.raises(ValueError, match="different specifications"):
        save_candidate_result(scored, "other", runners, code_root)


def test_legacy_leaderboard_with_holdout_metrics_fails_loudly(
    sealed: Path, code_root: Path
) -> None:
    (sealed / "results.json").write_text(
        json.dumps([{"candidate_id": "old", "metrics": {"test": {"auc": 0.9}}}])
    )
    with pytest.raises(ValidationError):
        save_candidate_result(sealed, "candidate-a", {"candidate-a": fitted}, code_root)


def test_a_new_run_before_any_read_withdraws_the_selection(scored: Path, code_root: Path) -> None:
    freeze_candidate(scored, "candidate-a", "label")
    save_candidate_result(scored, "candidate-a", {"candidate-a": fitted}, code_root)
    assert not (scored / "work/selection.json").exists()


def test_init_seeds_the_layout_and_keeps_edits_without_force(tmp_path: Path) -> None:
    experiment = init_experiment_dir(tmp_path / "new", ExperimentSeed("new"))
    assert {path.name for path in experiment.iterdir()} >= {
        "experiment.md",
        "research_journal.md",
        "research_sources.md",
        "results.json",
        "outputs",
        "work",
        "scripts",
    }
    (experiment / "experiment.md").write_text("edited")
    init_experiment_dir(experiment, ExperimentSeed("new"))
    assert (experiment / "experiment.md").read_text() == "edited"
    init_experiment_dir(experiment, ExperimentSeed("new"), force=True)
    assert (experiment / "experiment.md").read_text() != "edited"


@pytest.mark.parametrize("name", ["lazy_helper.py", "settings.json"])
def test_any_file_in_the_experiment_package_makes_evidence_stale(
    scored: Path, code_root: Path, name: str
) -> None:
    selected = load_results(scored)[0]
    assert run_evidence_staleness(scored, selected) == []
    (code_root / name).write_text("1\n")
    assert "Code or dependencies changed" in run_evidence_staleness(scored, selected)[0]
    freeze_candidate(scored, "candidate-a", "label")


def test_declared_config_file_is_fingerprinted(
    sealed: Path, code_root: Path, tmp_path: Path
) -> None:
    config = tmp_path / "params.json"
    config.write_text("{}")
    save_candidate_result(sealed, "candidate-a", {"candidate-a": fitted}, code_root, [config])
    selected = load_results(sealed)[0]
    assert run_evidence_staleness(sealed, selected) == []
    config.write_text('{"a": 1}')
    assert "Code or dependencies changed" in run_evidence_staleness(sealed, selected)[0]


def test_force_refuses_to_erase_consumed_holdout_evidence(scored: Path, code_root: Path) -> None:
    freeze_candidate(scored, "candidate-a", "label")
    evaluate_final(scored, {"candidate-a": fitted})
    with pytest.raises(ValueError, match="consumed"):
        init_experiment_dir(scored, ExperimentSeed("experiment"), force=True)
