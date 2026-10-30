"""Each test reproduces a scenario in which a used holdout was read again or left no record."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_ml_loop import final_evaluation
from agentic_ml_loop.candidate_result import CandidateRunner
from agentic_ml_loop.cli.main import main
from agentic_ml_loop.completion import (
    CompletionRecord,
    ResearchReview,
    check_completion,
    evidence_file,
)
from agentic_ml_loop.final_evaluation import evaluate_final, freeze_candidate, reseal_holdout
from agentic_ml_loop.holdout import (
    consumed_holdout_count,
    load_search_splits,
    read_holdout_seal,
    seal_holdout,
    sealed_file_path,
)
from agentic_ml_loop.io import write_json
from agentic_ml_loop.ledger import entries_for, read_ledger, store_root
from agentic_ml_loop.paths import ATTEMPT_PATH, FINAL_PATH
from agentic_ml_loop.runner import ExperimentSeed, init_experiment_dir, save_candidate_result
from agentic_ml_loop.splits import HoldoutSplits, temporal_split
from tests.conftest import SPEC, fitted, rows

RUNNERS: dict[str, CandidateRunner] = {"candidate-a": fitted}


def _experiment(path: Path, code_root: Path, seed: int = 8) -> Path:
    init_experiment_dir(path, ExperimentSeed(path.name))
    (path / "experiment.md").write_text(SPEC)
    seal_holdout(path, *temporal_split(rows(seed=seed), "when"))
    save_candidate_result(path, "candidate-a", RUNNERS, code_root)
    return path


def _new_rows(seed: int = 99) -> pd.DataFrame:
    return rows(count=12, seed=seed, start="2026-06-01")


def _freeze_and_evaluate(path: Path):
    freeze_candidate(path, "candidate-a", "label")
    return evaluate_final(path, RUNNERS)


def test_same_file_in_a_second_experiment_is_refused(tmp_path: Path, code_root: Path) -> None:
    first = _experiment(tmp_path / "x", code_root)
    second = _experiment(tmp_path / "y", code_root)
    assert read_holdout_seal(first).holdout_sha256 == read_holdout_seal(second).holdout_sha256
    _freeze_and_evaluate(first)
    freeze_candidate(second, "candidate-a", "label")
    with pytest.raises(ValueError, match="already evaluated by experiment x"):
        evaluate_final(second, RUNNERS)
    assert not (second / FINAL_PATH).exists()
    assert not (second / ATTEMPT_PATH).exists()


def test_crash_after_the_ledger_entry_costs_a_reseal_not_a_read(scored: Path, monkeypatch) -> None:
    freeze_candidate(scored, "candidate-a", "label")
    calls: list[str] = []

    def crash(*_args: object) -> None:
        calls.append("write")
        raise RuntimeError("crash before the attempt file")

    with monkeypatch.context() as patch:
        patch.setattr(final_evaluation, "write_json", crash)
        with pytest.raises(RuntimeError, match="crash"):
            evaluate_final(scored, RUNNERS)
    assert calls == ["write"]
    assert [entry.event for entry in read_ledger()] == ["evaluated"]
    with pytest.raises(ValueError, match="already evaluated"):
        evaluate_final(scored, RUNNERS)
    reseal_holdout(scored, _new_rows(), "The first evaluation crashed.")
    save_candidate_result(scored, "candidate-a", RUNNERS, scored.parent / "candidate_code")
    assert _freeze_and_evaluate(scored).prior_holdouts == 1


def test_scores_come_after_the_ledger_and_the_attempt(scored: Path) -> None:
    freeze_candidate(scored, "candidate-a", "label")
    seen: list[tuple[int, bool]] = []

    def runner(splits):
        result = fitted(splits)

        def score_rows(frame: pd.DataFrame) -> dict[str, float]:
            seen.append((len(read_ledger()), (scored / ATTEMPT_PATH).is_file()))
            return result.score_rows(frame)

        return type(result)(result=result.result, score_rows=score_rows)

    evaluate_final(scored, {"candidate-a": runner})
    assert seen == [(1, True)]


def test_killed_reseal_leaves_no_readable_holdout(
    scored: Path, code_root: Path, monkeypatch
) -> None:
    freeze_candidate(scored, "candidate-a", "label")
    evaluate_final(scored, RUNNERS)
    before = read_holdout_seal(scored)

    def kill(*_args: object) -> None:
        raise RuntimeError("killed before commit_seal")

    with monkeypatch.context() as patch:
        patch.setattr(final_evaluation, "commit_seal", kill)
        with pytest.raises(RuntimeError, match="killed"):
            reseal_holdout(scored, _new_rows(), "Retiring.")

    assert read_holdout_seal(scored) == before
    save_candidate_result(scored, "candidate-a", RUNNERS, code_root)
    freeze_candidate(scored, "candidate-a", "label")
    with pytest.raises(ValueError, match="already evaluated"):
        evaluate_final(scored, RUNNERS)

    new = reseal_holdout(scored, _new_rows(), "Retiring.")
    assert read_holdout_seal(scored) == new
    assert consumed_holdout_count(scored) == 1
    assert [e.event for e in entries_for(before.holdout_sha256)] == ["evaluated", "retired"]
    assert entries_for(before.holdout_sha256)[1].provenance_note == "Retiring."


def test_retired_unread_holdout_cannot_be_read(scored: Path, code_root: Path, monkeypatch) -> None:
    def kill(*_args: object) -> None:
        raise RuntimeError("killed")

    with monkeypatch.context() as patch:
        patch.setattr(final_evaluation, "seal_replacement_rows", kill)
        with pytest.raises(RuntimeError):
            reseal_holdout(scored, _new_rows(), "Retiring.")
    save_candidate_result(scored, "candidate-a", RUNNERS, code_root)
    freeze_candidate(scored, "candidate-a", "label")
    with pytest.raises(ValueError, match="already retired"):
        evaluate_final(scored, RUNNERS)


def test_returning_to_an_earlier_file_is_refused(scored: Path) -> None:
    original = pd.read_parquet(sealed_file_path(read_holdout_seal(scored)))
    reseal_holdout(scored, _new_rows(), "First.")
    with pytest.raises(ValueError, match="already retired"):
        reseal_holdout(scored, original, "Back to the first file.")


def _write_review_and_record(path: Path, **fields: object) -> CompletionRecord:
    (path / "outputs/report.md").write_text("Report.")
    (path / "outputs/reproduce.md").write_text("Commands.")
    refs = {
        "specification": evidence_file(path, "experiment.md"),
        "results": evidence_file(path, "results.json"),
        "report": evidence_file(path, "outputs/report.md"),
        "evaluation": evidence_file(path, FINAL_PATH),
        "reproduction": evidence_file(path, "outputs/reproduce.md"),
    }
    review = ResearchReview(
        reviewer="reviewer",
        verdict="accepted",
        specification_sha256=refs["specification"].sha256,
        results_sha256=refs["results"].sha256,
        report_sha256=refs["report"].sha256,
        evaluation_sha256=refs["evaluation"].sha256,
        findings=[],
        limitations=["Synthetic."],
    )
    write_json(path / "outputs/review.json", review.model_dump())
    return CompletionRecord.model_validate(
        {
            "outcome": "no_demonstrated_improvement",
            "conclusion": "Kept.",
            "limitations": ["Synthetic."],
            "selected_candidate_id": "candidate-a",
            "baseline_candidate_id": "candidate-a",
            "comparison_candidate_id": "other",
            "review": evidence_file(path, "outputs/review.json"),
            **refs,
            **fields,
        }
    )


def test_hand_written_final_evidence_fails_completion(scored: Path) -> None:
    selection = freeze_candidate(scored, "candidate-a", "label")
    seal = read_holdout_seal(scored)
    selection_sha = evidence_file(scored, "work/selection.json").sha256
    attempt = final_evaluation.FinalAttempt(
        selected_run_id=selection.selected_result.run_id,
        selection_sha256=selection_sha,
        holdout=seal,
        started_at="2026-01-01",
    )
    write_json(scored / ATTEMPT_PATH, attempt.model_dump(mode="json"))
    forged = {
        "selected_run_id": selection.selected_result.run_id,
        "selection_sha256": selection_sha,
        "attempt_sha256": evidence_file(scored, ATTEMPT_PATH).sha256,
        "holdout": seal.model_dump(),
        "completed_at": "2026-01-01",
        "metrics": {"auc": 0.999},
        "selection_metrics": selection.selected_result.metrics.model_dump(),
        "prior_holdouts": 0,
        "holdout_target_concern": None,
        "overlap": {
            "key_columns": [],
            "train_validation_exact_rows": 0,
            "search_holdout_exact_rows": 0,
            "train_validation_keys": None,
            "search_holdout_keys": None,
        },
        "limitations": [],
    }
    write_json(scored / FINAL_PATH, forged)
    record = _write_review_and_record(scored)
    errors = check_completion(scored, record)
    assert any("ledger has no evaluation" in error for error in errors)


def test_real_final_evidence_chains_to_the_ledger(scored: Path) -> None:
    final = _freeze_and_evaluate(scored)
    entry = entries_for(final.holdout.holdout_sha256)[0]
    assert entry.experiment_uid == final.holdout.experiment_uid
    assert entry.attempt_sha256 == final.attempt_sha256
    assert entry.attempt_sha256 == evidence_file(scored, ATTEMPT_PATH).sha256


def test_overlap_is_reported_and_completion_requires_acknowledgement(
    tmp_path: Path, code_root: Path
) -> None:
    path = tmp_path / "overlapping"
    init_experiment_dir(path, ExperimentSeed("overlapping"))
    (path / "experiment.md").write_text(SPEC + "\nkey_columns: when\n")
    source = rows(count=400)
    search, holdout = temporal_split(source, "when")
    leaked = pd.concat([search.train.head(150), holdout.holdout.head(10)], ignore_index=True)
    seal_holdout(path, search, HoldoutSplits(holdout=leaked))
    save_candidate_result(path, "candidate-a", RUNNERS, code_root)
    final = _freeze_and_evaluate(path)
    assert final.overlap.search_holdout_exact_rows == 150
    assert final.overlap.search_holdout_keys == 150
    assert final.overlap.key_columns == ["when"]
    errors = check_completion(path, _write_review_and_record(path))
    assert any("reports shared rows" in error for error in errors)
    assert not any(
        "reports shared rows" in error
        for error in check_completion(
            path, _write_review_and_record(path, acknowledged_overlap="Known duplicates.")
        )
    )


def test_overlap_report_says_when_a_cast_makes_rows_incomparable(
    tmp_path: Path, code_root: Path
) -> None:
    path = tmp_path / "cast"
    init_experiment_dir(path, ExperimentSeed("cast"))
    (path / "experiment.md").write_text(SPEC)
    search, holdout = temporal_split(rows(count=100), "when")
    cast = holdout.holdout.assign(label=holdout.holdout["label"].astype(float))
    seal_holdout(path, search, HoldoutSplits(holdout=cast))
    save_candidate_result(path, "candidate-a", RUNNERS, code_root)
    assert _freeze_and_evaluate(path).overlap.search_holdout_exact_rows is None


def test_freeze_reports_drift_without_reading_the_holdout(sealed: Path, code_root: Path) -> None:
    path = sealed / "data/validation.parquet"
    pd.read_parquet(path).assign(label=1).to_parquet(path, index=False)
    save_candidate_result(sealed, "candidate-a", RUNNERS, code_root)
    sealed_file_path(read_holdout_seal(sealed)).unlink()
    audit = freeze_candidate(sealed, "candidate-a", "label").data_audit
    assert any("validation target rate differs" in concern for concern in audit.concerns)


def test_rows_inside_the_experiment_are_refused(scored: Path, capsys) -> None:
    inside = scored / "work" / "nested" / "rows.parquet"
    inside.parent.mkdir(parents=True)
    _new_rows().to_parquet(inside, index=False)
    code = main(["reseal", "--experiment", str(scored), "--rows", str(inside), "--note", "x"])
    assert code == 1
    assert "outside the experiment" in capsys.readouterr().err
    linked = scored.parent / "link.parquet"
    linked.symlink_to(inside)
    assert main(["reseal", "--experiment", str(scored), "--rows", str(linked), "--note", "x"]) == 1
    assert read_ledger() == []


def test_concurrent_seals_cannot_mix_datasets(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "raced"
    path.mkdir()
    start = threading.Barrier(2)
    outcomes: list[object] = []

    def seal(seed: int) -> None:
        start.wait()
        try:
            outcomes.append(seal_holdout(path, *temporal_split(rows(seed=seed), "when")))
        except ValueError as exc:
            outcomes.append(exc)

    threads = [threading.Thread(target=seal, args=(seed,)) for seed in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    won = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    assert len(won) == 1
    winner = next(
        seed
        for seed in (1, 2)
        if temporal_split(rows(seed=seed), "when")[0]
        .validation["signal"]
        .equals(load_search_splits(path).validation["signal"])
    )
    train = temporal_split(rows(seed=winner), "when")[0].train
    assert np.array_equal(load_search_splits(path).train["signal"], train["signal"])
    held = temporal_split(rows(seed=winner), "when")[1].holdout
    sealed_rows = pd.read_parquet(sealed_file_path(read_holdout_seal(path)))
    assert np.array_equal(sealed_rows["signal"], held["signal"])


def test_six_fresh_seeds_all_evaluate(tmp_path: Path, code_root: Path) -> None:
    for seed in range(1, 7):
        final = _freeze_and_evaluate(_experiment(tmp_path / f"run{seed}", code_root, seed=seed))
        assert final.prior_holdouts == 0
    assert len(read_ledger()) == 6


def test_ledger_file_lives_outside_workspaces(scored: Path) -> None:
    _freeze_and_evaluate(scored)
    ledger = store_root() / "ledger.jsonl"
    assert json.loads(ledger.read_text().splitlines()[0])["event"] == "evaluated"
    assert not ledger.resolve().is_relative_to(scored.resolve())


def test_force_never_deletes_holdout_evidence(scored: Path) -> None:
    reseal_holdout(scored, _new_rows(), "First retirement.")
    seal = read_holdout_seal(scored)
    init_experiment_dir(scored, ExperimentSeed("experiment"), force=True)
    assert read_holdout_seal(scored) == seal
    assert consumed_holdout_count(scored) == 1
    assert (scored / "work/holdout_history/0001/consumed.json").is_file()


def test_freeze_requires_the_target_column(scored: Path) -> None:
    with pytest.raises(ValueError, match="Target column"):
        freeze_candidate(scored, "candidate-a", "absent")


def test_lost_ledger_fails_closed_for_this_experiments_history(scored: Path) -> None:
    reseal_holdout(scored, _new_rows(), "First retirement.")
    (store_root() / "ledger.jsonl").unlink()
    with pytest.raises(ValueError, match="ledger lacks"):
        reseal_holdout(scored, _new_rows(seed=100), "Second retirement.")


def test_corrupt_ledger_fails_closed(scored: Path) -> None:
    reseal_holdout(scored, _new_rows(), "First retirement.")
    (store_root() / "ledger.jsonl").write_text("not json\n")
    with pytest.raises(ValueError):
        reseal_holdout(scored, _new_rows(seed=100), "Second retirement.")


def test_unwritable_store_fails_closed(sealed: Path, tmp_path: Path, monkeypatch) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("XDG_DATA_HOME", str(blocker))
    with pytest.raises(OSError):
        reseal_holdout(sealed, _new_rows(), "Store unreachable.")
