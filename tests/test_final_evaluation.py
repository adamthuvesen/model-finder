from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from agentic_ml_loop.candidate_result import CandidateRunner, FittedCandidate
from agentic_ml_loop.final_evaluation import evaluate_final, freeze_candidate, reseal_holdout
from agentic_ml_loop.ledger import read_ledger
from agentic_ml_loop.paths import ATTEMPT_PATH, FINAL_PATH, SELECTION_PATH
from agentic_ml_loop.results import load_results
from agentic_ml_loop.runner import save_candidate_result
from agentic_ml_loop.splits import SearchSplits
from agentic_ml_loop.validation import validate_experiment
from tests.conftest import fitted

RUNNERS: dict[str, CandidateRunner] = {"candidate-a": fitted}


def freeze(path: Path):
    return freeze_candidate(path, "candidate-a", "label")


def test_freeze_is_idempotent_and_final_evidence_stays_off_the_leaderboard(scored: Path) -> None:
    results_before = (scored / "results.json").read_bytes()
    selection = freeze(scored)
    assert freeze(scored) == selection
    final = evaluate_final(scored, RUNNERS)
    assert set(final.metrics) == {"auc"}
    assert final.selection_metrics == selection.selected_result.metrics
    assert (scored / "results.json").read_bytes() == results_before
    assert (scored / FINAL_PATH).is_file()


def test_completed_evaluation_returns_evidence_without_refitting(scored: Path) -> None:
    freeze(scored)
    final = evaluate_final(scored, RUNNERS)

    def forbidden(_splits: SearchSplits) -> FittedCandidate:
        raise AssertionError("Cached final evaluation reran the model")

    assert evaluate_final(scored, {"candidate-a": forbidden}) == final


def test_freeze_requires_a_known_candidate(scored: Path) -> None:
    with pytest.raises(ValueError, match="Expected one candidate"):
        freeze_candidate(scored, "absent", "label")


def _mutate_results(path: Path) -> None:
    (path / "results.json").write_text((path / "results.json").read_text() + "\n")


def _mutate_spec(path: Path) -> None:
    (path / "experiment.md").write_text((path / "experiment.md").read_text() + "\nmore\n")


def _mutate_selection(path: Path) -> None:
    (path / SELECTION_PATH).write_text("{}")


@pytest.mark.parametrize("mutate", [_mutate_results, _mutate_selection])
def test_stale_selection_prevents_the_final_fit(scored: Path, mutate) -> None:
    freeze(scored)
    mutate(scored)

    def forbidden(_splits: SearchSplits) -> FittedCandidate:
        raise AssertionError("Stale selection reached the model")

    with pytest.raises(ValueError):
        evaluate_final(scored, {"candidate-a": forbidden})
    assert not (scored / ATTEMPT_PATH).exists()


def test_changed_code_or_spec_is_recorded_and_does_not_stop_the_final(
    scored: Path, code_root: Path
) -> None:
    freeze(scored)
    (code_root / "candidate_module.py").write_text("VERSION = 2\n")
    _mutate_spec(scored)
    final = evaluate_final(scored, RUNNERS)
    assert any("Code or dependencies changed" in text for text in final.limitations)
    assert any("specification changed" in text for text in final.limitations)


def test_changed_search_rows_prevent_the_final_fit(scored: Path) -> None:
    freeze(scored)
    train_path = scored / "data/train.parquet"
    pd.read_parquet(train_path).assign(signal=0.0).to_parquet(train_path, index=False)
    with pytest.raises(ValueError, match="Search data changed"):
        evaluate_final(scored, RUNNERS)
    assert not (scored / ATTEMPT_PATH).exists()


def test_interrupted_attempt_fails_loudly_on_retry(scored: Path) -> None:
    freeze(scored)

    def crashes(splits: SearchSplits) -> FittedCandidate:
        def explode(_rows: pd.DataFrame) -> dict[str, float]:
            raise RuntimeError("interrupted")

        return FittedCandidate(result=fitted(splits).result, score_rows=explode)

    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate_final(scored, {"candidate-a": crashes})
    assert (scored / ATTEMPT_PATH).is_file()
    with pytest.raises(ValueError, match="did not finish"):
        evaluate_final(scored, RUNNERS)


def test_rerun_that_changes_the_result_is_rejected(scored: Path) -> None:
    freeze(scored)

    def different(splits: SearchSplits) -> FittedCandidate:
        original = fitted(splits)
        changed = original.result.model_copy(update={"feature_set": "changed"})
        return FittedCandidate(result=changed, score_rows=original.score_rows)

    with pytest.raises(ValueError, match="feature_set"):
        evaluate_final(scored, {"candidate-a": different})
    assert not (scored / FINAL_PATH).exists()


def test_final_evidence_without_an_attempt_is_rejected(scored: Path) -> None:
    freeze(scored)
    (scored / FINAL_PATH).write_text("{}")
    with pytest.raises(ValueError, match="no recorded evaluation attempt"):
        evaluate_final(scored, RUNNERS)


def test_tampered_final_evidence_is_rejected(scored: Path) -> None:
    freeze(scored)
    final = evaluate_final(scored, RUNNERS)
    payload = final.model_dump(mode="json")
    payload["selected_run_id"] = "other"
    (scored / FINAL_PATH).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match"):
        evaluate_final(scored, RUNNERS)


def test_unavailable_frozen_candidate_is_reported(scored: Path) -> None:
    freeze(scored)
    with pytest.raises(ValueError, match="not available"):
        evaluate_final(scored, {})


def test_results_stay_validated_after_a_freeze(scored: Path) -> None:
    freeze(scored)
    assert load_results(scored)[0].candidate_id == "candidate-a"


def _new_rows(count: int = 12, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "when": pd.date_range("2026-06-01", periods=count),
            "signal": rng.normal(size=count),
            "label": np.resize([0, 1], count),
        }
    )


def test_reseal_retires_the_read_holdout_and_reopens_search(scored: Path, code_root: Path) -> None:
    run_ids = [result.run_id for result in load_results(scored)]
    freeze(scored)
    first = evaluate_final(scored, RUNNERS)
    assert first.prior_holdouts == 0

    seal = reseal_holdout(scored, _new_rows(), "The first holdout was read.")

    assert seal.holdout_rows == 12
    consumed = json.loads((scored / "work/holdout_history/0001/consumed.json").read_text())
    assert consumed["holdout"] == first.holdout.model_dump()
    assert [(e.event, e.holdout_sha256) for e in read_ledger()] == [
        ("evaluated", first.holdout.holdout_sha256),
        ("finalized", first.holdout.holdout_sha256),
        ("retired", first.holdout.holdout_sha256),
    ]
    assert consumed["exploratory_run_ids"] == run_ids
    assert consumed["provenance_note"] == "The first holdout was read."
    assert (scored / "work/holdout_history/0001/final_evaluation.json").is_file()
    assert not (scored / FINAL_PATH).exists()
    assert not (scored / SELECTION_PATH).exists()
    assert any("holdouts retired" in message for message in validate_experiment(scored))

    save_candidate_result(scored, "candidate-a", RUNNERS, code_root)
    freeze(scored)
    second = evaluate_final(scored, RUNNERS)
    assert second.prior_holdouts == 1
    assert second.metrics != first.metrics


def test_reseal_requires_matching_columns_and_a_note(scored: Path) -> None:
    with pytest.raises(ValueError, match="same columns"):
        reseal_holdout(scored, _new_rows().drop(columns="signal"), "Wrong shape.")
    with pytest.raises(ValidationError):
        reseal_holdout(scored, _new_rows(), "   ")
    assert read_ledger() == []
