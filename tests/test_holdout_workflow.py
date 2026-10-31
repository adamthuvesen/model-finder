"""Each test reproduces a scenario in which a used holdout was read again or left no record."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_finder import final_evaluation, reseal
from model_finder.candidate_result import CandidateRunner
from model_finder.completion import (
    CompletionRecord,
    ResearchReview,
    assess_completion,
    check_completion,
    evidence_file,
)
from model_finder.data_audit import OverlapReport
from model_finder.final_evaluation import evaluate_final, freeze_candidate
from model_finder.holdout import (
    consumed_holdout_count,
    load_search_splits,
    read_holdout_seal,
    seal_holdout,
    sealed_file_path,
)
from model_finder.io import write_json
from model_finder.ledger import entries_for, read_ledger, store_root
from model_finder.paths import ATTEMPT_PATH, FINAL_PATH
from model_finder.reseal import reseal_holdout
from model_finder.run_evidence import (
    _dependencies_sha256,
    code_fingerprint,
    recorded_fingerprint_is_current,
)
from model_finder.runner import ExperimentSeed, init_experiment_dir, save_candidate_result
from model_finder.splits import HoldoutSplits, SearchSplits, temporal_split
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
        patch.setattr(reseal, "commit_seal", kill)
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
    assert [e.event for e in entries_for(before.holdout_sha256)] == [
        "evaluated",
        "finalized",
        "retired",
    ]
    assert entries_for(before.holdout_sha256)[2].provenance_note == "Retiring."


def test_retired_unread_holdout_cannot_be_read(scored: Path, code_root: Path, monkeypatch) -> None:
    def kill(*_args: object) -> None:
        raise RuntimeError("killed")

    with monkeypatch.context() as patch:
        patch.setattr(reseal, "_write_history", kill)
        with pytest.raises(RuntimeError):
            reseal_holdout(scored, _new_rows(), "Retiring.")
    save_candidate_result(scored, "candidate-a", RUNNERS, code_root)
    freeze_candidate(scored, "candidate-a", "label")
    with pytest.raises(ValueError, match="already retired"):
        evaluate_final(scored, RUNNERS)


def test_returning_to_an_earlier_file_needs_a_reason_to_read_it(scored: Path) -> None:
    original = pd.read_parquet(sealed_file_path(read_holdout_seal(scored)))
    reseal_holdout(scored, _new_rows(), "First.")
    reseal_holdout(scored, original, "Back to the first file.")
    freeze_candidate(scored, "candidate-a", "label")
    with pytest.raises(ValueError, match="allow-reread"):
        evaluate_final(scored, RUNNERS)


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


def test_final_evidence_the_ledger_never_saw_is_labelled_unverified(scored: Path) -> None:
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
        "exploratory_reasons": [],
        "holdout_target_concern": None,
        "overlap": {
            "key_columns": [],
            "train_validation_exact_rows": 0,
            "search_holdout_exact_rows": 0,
            "train_validation_keys": None,
            "search_holdout_keys": None,
            "not_comparable": [],
            "holdout_rows": seal.holdout_rows,
        },
        "limitations": [],
    }
    write_json(scored / FINAL_PATH, forged)
    status = assess_completion(
        scored, _write_review_and_record(scored, comparison_candidate_id=None)
    )
    assert status.errors == []
    assert status.label == "complete, exploratory"
    assert any("no record of this read" in reason for reason in status.exploratory)


def test_real_final_evidence_chains_to_the_ledger(scored: Path) -> None:
    final = _freeze_and_evaluate(scored)
    entry = entries_for(final.holdout.holdout_sha256)[0]
    assert entry.experiment_uid == final.holdout.experiment_uid
    assert entry.attempt_sha256 == final.attempt_sha256
    assert entry.attempt_sha256 == evidence_file(scored, ATTEMPT_PATH).sha256


def test_large_overlap_needs_a_one_line_reason(tmp_path: Path, code_root: Path) -> None:
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
    assert any("--acknowledged-overlap" in error for error in errors)
    status = assess_completion(
        path,
        _write_review_and_record(
            path, acknowledged_overlap="Known duplicates.", comparison_candidate_id=None
        ),
    )
    assert status.errors == []
    assert status.label == "complete, exploratory"


def test_freeze_reports_drift_without_reading_the_holdout(sealed: Path, code_root: Path) -> None:
    path = sealed / "data/validation.parquet"
    pd.read_parquet(path).assign(label=1).to_parquet(path, index=False)
    save_candidate_result(sealed, "candidate-a", RUNNERS, code_root)
    sealed_file_path(read_holdout_seal(sealed)).unlink()
    audit = freeze_candidate(sealed, "candidate-a", "label").data_audit
    assert any("validation target rate differs" in concern for concern in audit.concerns)


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
    assert sum(entry.event == "evaluated" for entry in read_ledger()) == 6


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


def test_lost_ledger_does_not_stop_a_reseal(scored: Path) -> None:
    reseal_holdout(scored, _new_rows(), "First retirement.")
    (store_root() / "ledger.jsonl").unlink()
    assert reseal_holdout(scored, _new_rows(seed=100), "Second retirement.")


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


def _kill(*_args: object) -> None:
    raise RuntimeError("killed")


def _completion_errors(path: Path, **fields: object) -> list[str]:
    return check_completion(path, _write_review_and_record(path, **fields))


def test_forged_final_after_a_kill_blocks_completion(scored: Path) -> None:
    freeze_candidate(scored, "candidate-a", "label")

    def killed(splits):
        result = fitted(splits)

        def crash(_rows: pd.DataFrame) -> dict[str, float]:
            raise RuntimeError("killed after the attempt")

        return type(result)(result=result.result, score_rows=crash)

    with pytest.raises(RuntimeError):
        evaluate_final(scored, {"candidate-a": killed})
    selection = final_evaluation.FrozenSelection.model_validate_json(
        (scored / "work/selection.json").read_text()
    )
    attempt = final_evaluation.FinalAttempt.model_validate_json((scored / ATTEMPT_PATH).read_text())
    forged = final_evaluation.FinalEvaluation(
        selected_run_id=attempt.selected_run_id,
        selection_sha256=attempt.selection_sha256,
        attempt_sha256=evidence_file(scored, ATTEMPT_PATH).sha256,
        holdout=attempt.holdout,
        completed_at="x",
        metrics={"auc": 0.999},
        selection_metrics=selection.selected_result.metrics,
        prior_holdouts=0,
        exploratory_reasons=[],
        holdout_target_concern=None,
        overlap=OverlapReport(
            key_columns=[],
            train_validation_exact_rows=0,
            search_holdout_exact_rows=0,
            train_validation_keys=None,
            search_holdout_keys=None,
            not_comparable=[],
            holdout_rows=attempt.holdout.holdout_rows,
        ),
        limitations=[],
    )
    write_json(scored / FINAL_PATH, forged.model_dump(mode="json"))
    assert any("edited or forged" in error for error in _completion_errors(scored))


def test_editing_a_genuine_final_fails_completion(scored: Path) -> None:
    _freeze_and_evaluate(scored)
    payload = json.loads((scored / FINAL_PATH).read_text())
    payload["metrics"]["auc"] = 0.999
    write_json(scored / FINAL_PATH, payload)
    assert any("edited or forged" in error for error in _completion_errors(scored))


def test_resealing_the_read_live_holdout_changes_nothing(scored: Path) -> None:
    seal = read_holdout_seal(scored)
    same_rows = pd.read_parquet(sealed_file_path(seal))
    _freeze_and_evaluate(scored)
    events = [entry.event for entry in read_ledger()]
    with pytest.raises(ValueError, match="already read"):
        reseal_holdout(scored, same_rows, "Same rows.")
    assert read_holdout_seal(scored) == seal
    assert [entry.event for entry in read_ledger()] == events


def test_retry_after_commit_is_a_no_op_and_keeps_fresh_rows(scored: Path, monkeypatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(reseal, "remove_other_sealed_files", _kill)
        with pytest.raises(RuntimeError):
            reseal_holdout(scored, _new_rows(), "Retiring.")
    committed = read_holdout_seal(scored)
    assert reseal_holdout(scored, _new_rows(), "Retiring.") == committed
    assert entries_for(committed.holdout_sha256) == []
    assert consumed_holdout_count(scored) == 1
    assert sealed_file_path(committed).is_file()
    assert len(list(sealed_file_path(committed).parent.glob("*.parquet"))) == 1


@pytest.mark.parametrize("kill_point", ["_write_history", "place_staged_file", "commit_seal", None])
def test_prior_count_is_the_same_at_every_kill_point(
    scored: Path, monkeypatch, kill_point: str | None
) -> None:
    _freeze_and_evaluate(scored)
    with monkeypatch.context() as patch:
        if kill_point:
            patch.setattr(reseal, kill_point, _kill)
            with pytest.raises(RuntimeError):
                reseal_holdout(scored, _new_rows(), "Retiring.")
        else:
            reseal_holdout(scored, _new_rows(), "Retiring.")
    assert consumed_holdout_count(scored) == 1


def test_deleting_history_does_not_lower_the_count(scored: Path) -> None:
    reseal_holdout(scored, _new_rows(), "Retiring.")
    shutil.rmtree(scored / "work/holdout_history")
    assert consumed_holdout_count(scored) == 1


def test_kill_before_history_then_second_reseal_counts_two(scored: Path, monkeypatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(reseal, "_write_history", _kill)
        with pytest.raises(RuntimeError):
            reseal_holdout(scored, _new_rows(), "First.")
    reseal_holdout(scored, _new_rows(), "First.")
    reseal_holdout(scored, _new_rows(seed=100), "Second.")
    assert consumed_holdout_count(scored) == 2


def test_holdout_equal_to_train_with_one_nan_is_counted(tmp_path: Path, code_root: Path) -> None:
    path = tmp_path / "nan"
    init_experiment_dir(path, ExperimentSeed("nan"))
    (path / "experiment.md").write_text(SPEC)
    search, _ = temporal_split(rows(count=100), "when")
    copied = search.train.copy()
    copied["label"] = copied["label"].astype(float)
    copied.loc[copied.index[0], "label"] = float("nan")
    seal_holdout(path, search, HoldoutSplits(holdout=copied))
    save_candidate_result(path, "candidate-a", RUNNERS, code_root)
    overlap = _freeze_and_evaluate(path).overlap
    assert overlap.search_holdout_exact_rows == len(copied) - 1


def test_a_comparison_that_cannot_run_warns(tmp_path: Path, code_root: Path) -> None:
    path = tmp_path / "text"
    init_experiment_dir(path, ExperimentSeed("text"))
    (path / "experiment.md").write_text(SPEC)
    search, holdout = temporal_split(rows(count=100), "when")
    seal_holdout(path, search, HoldoutSplits(holdout=holdout.holdout.astype({"signal": str})))
    save_candidate_result(path, "candidate-a", _text_tolerant_runners(), code_root)
    freeze_candidate(path, "candidate-a", "label")
    final = evaluate_final(path, _text_tolerant_runners())
    assert final.overlap.not_comparable == ["search_holdout_exact_rows"]
    status = assess_completion(path, _write_review_and_record(path, comparison_candidate_id=None))
    assert status.errors == []
    assert any("could not run" in warning for warning in status.warnings)


def _text_tolerant_runners() -> dict[str, CandidateRunner]:
    def runner(splits):
        result = fitted(splits)
        return type(result)(
            result=result.result, score_rows=lambda frame: {"auc": float(len(frame))}
        )

    return {"candidate-a": runner}


def test_key_misspelled_after_freeze_reports_instead_of_losing_the_read(scored: Path) -> None:
    freeze_candidate(scored, "candidate-a", "label")
    (scored / "experiment.md").write_text(SPEC + "\nkey_columns: wehn\n")
    final = evaluate_final(scored, RUNNERS)
    assert "train_validation_keys" in final.overlap.not_comparable


def test_misspelled_key_fails_the_freeze_before_any_read(sealed: Path, code_root: Path) -> None:
    (sealed / "experiment.md").write_text(SPEC + "\nkey_columns: wehn\n")
    save_candidate_result(sealed, "candidate-a", RUNNERS, code_root)
    with pytest.raises(ValueError, match="wehn"):
        freeze_candidate(sealed, "candidate-a", "label")
    assert read_ledger() == []


def test_duplicate_and_nan_keys(tmp_path: Path, code_root: Path) -> None:
    path = tmp_path / "keys"
    init_experiment_dir(path, ExperimentSeed("keys"))
    (path / "experiment.md").write_text(SPEC + "\nkey_columns: account, account\n")
    search, holdout = temporal_split(rows(count=100), "when")
    with_keys = SearchSplits(
        train=search.train.assign(account=np.nan),
        validation=search.validation.assign(account=np.nan),
    )
    seal_holdout(path, with_keys, HoldoutSplits(holdout=holdout.holdout.assign(account=np.nan)))
    save_candidate_result(path, "candidate-a", RUNNERS, code_root)
    overlap = _freeze_and_evaluate(path).overlap
    assert overlap.key_columns == ["account"]
    assert overlap.search_holdout_keys == 0


def test_swapped_search_rows_fail_completion(scored: Path) -> None:
    _freeze_and_evaluate(scored)
    train = scored / "data/train.parquet"
    pd.read_parquet(train).assign(signal=0.0).to_parquet(train, index=False)
    assert any("Search data changed" in error for error in _completion_errors(scored))


def test_copied_evidence_warns_in_the_copy(scored: Path) -> None:
    _freeze_and_evaluate(scored)
    copy = scored.parent / "copied"
    shutil.copytree(scored, copy)
    status = assess_completion(copy, _write_review_and_record(copy))
    assert any("was produced in" in warning for warning in status.warnings)


def test_corrupt_ledger_error_names_the_file_and_line(scored: Path) -> None:
    ledger = store_root() / "ledger.jsonl"
    reseal_holdout(scored, _new_rows(), "First.")
    ledger.write_text(ledger.read_text() + "not json\n")
    with pytest.raises(ValueError, match=rf"{ledger}.*line 2"):
        read_ledger()


def test_dependency_hash_ignores_stray_egg_info_on_the_path(tmp_path: Path, monkeypatch) -> None:
    before = _dependencies_sha256()
    egg = tmp_path / "stray.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text("Name: stray\nVersion: 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    _dependencies_sha256.cache_clear()
    assert _dependencies_sha256() == before


def test_symlinked_helper_directory_is_fingerprinted(tmp_path: Path) -> None:
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    (helpers / "helper.py").write_text("1\n")
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "linked").symlink_to(helpers)
    (root / "loop").symlink_to(root)
    experiment = tmp_path / "e"
    experiment.mkdir()
    recorded = code_fingerprint(experiment, root)
    assert recorded_fingerprint_is_current(experiment, recorded)
    (helpers / "helper.py").write_text("2\n")
    assert not recorded_fingerprint_is_current(experiment, recorded)


def _evaluated_in_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    code = repository / "candidate_code"
    code.mkdir()
    (code / "module.py").write_text("VERSION = 1\n")
    experiment = _experiment(repository / "experiments" / "run", code)
    runners: dict[str, CandidateRunner] = {
        "candidate-b": lambda splits: fitted(splits, "candidate-b")
    }
    save_candidate_result(experiment, "candidate-b", runners, code)
    _freeze_and_evaluate(experiment)
    return experiment


def _supported_errors(experiment: Path) -> list[str]:
    return _completion_errors(
        experiment,
        outcome="supported",
        baseline_candidate_id="candidate-b",
        comparison_candidate_id=None,
    )


def test_moving_the_whole_repository_keeps_valid_evidence(tmp_path: Path) -> None:
    experiment = _evaluated_in_repository(tmp_path)
    assert _supported_errors(experiment) == []
    moved = tmp_path / "elsewhere" / "renamed"
    moved.parent.mkdir()
    shutil.move(tmp_path / "repository", moved)
    assert _supported_errors(moved / "experiments" / "run") == []


def test_copy_to_a_sibling_directory_in_the_repository_warns(tmp_path: Path) -> None:
    experiment = _evaluated_in_repository(tmp_path)
    sibling = experiment.parent / "sibling"
    shutil.copytree(experiment, sibling)
    status = assess_completion(
        sibling,
        _write_review_and_record(
            sibling,
            outcome="supported",
            baseline_candidate_id="candidate-b",
            comparison_candidate_id=None,
        ),
    )
    assert status.errors == []
    assert any("was produced in" in warning for warning in status.warnings)


def test_completion_must_select_the_frozen_candidate(tmp_path: Path) -> None:
    experiment = _evaluated_in_repository(tmp_path)
    errors = _completion_errors(
        experiment,
        outcome="supported",
        selected_candidate_id="candidate-b",
        baseline_candidate_id="candidate-a",
        comparison_candidate_id=None,
    )
    assert errors == ["Completion selects candidate-b, but candidate-a was frozen"]
