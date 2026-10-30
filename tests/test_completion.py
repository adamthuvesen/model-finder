from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from model_finder.completion import (
    CompletionRecord,
    EvidenceFile,
    ResearchReview,
    assess_completion,
    check_completion,
    evidence_file,
    record_from_evidence,
    save_completion,
)
from model_finder.holdout import HoldoutSeal
from model_finder.io import write_json
from model_finder.ledger import LedgerEntry, record_retirement
from model_finder.runner import ExperimentSeed, init_experiment_dir
from tests.conftest import SPEC


def _references(experiment: Path) -> dict[str, EvidenceFile]:
    return {
        "specification": evidence_file(experiment, "experiment.md"),
        "results": evidence_file(experiment, "results.json"),
        "report": evidence_file(experiment, "outputs/report.md"),
        "evaluation": evidence_file(experiment, "outputs/evaluation.md"),
        "reproduction": evidence_file(experiment, "outputs/reproduce.md"),
    }


def _write_review(
    experiment: Path, verdict: Literal["accepted", "changes_requested"] = "accepted"
) -> EvidenceFile:
    refs = _references(experiment)
    review = ResearchReview(
        reviewer="independent test reviewer",
        verdict=verdict,
        specification_sha256=refs["specification"].sha256,
        results_sha256=refs["results"].sha256,
        report_sha256=refs["report"].sha256,
        evaluation_sha256=refs["evaluation"].sha256,
        findings=[],
        limitations=["Only the absence of evidence was reviewed."],
    )
    write_json(experiment / "outputs/review.json", review.model_dump())
    return evidence_file(experiment, "outputs/review.json")


def _write_outputs(experiment: Path) -> None:
    for name, text in (
        ("report.md", "Inconclusive: the declared data source is unavailable."),
        ("evaluation.md", "No estimates were produced. Data access failed."),
        ("reproduce.md", "Restore the declared dataset and rerun the baseline."),
    ):
        (experiment / "outputs" / name).write_text(text)


def _record(experiment: Path, **overrides: object) -> CompletionRecord:
    return CompletionRecord(
        **{
            "outcome": "inconclusive",
            "conclusion": "Available evidence cannot answer the question.",
            "limitations": ["No data was available."],
            "review": _write_review(experiment),
            **_references(experiment),
            **overrides,
        }
    )


@pytest.fixture
def reviewed(tmp_path: Path) -> tuple[Path, CompletionRecord]:
    experiment = init_experiment_dir(tmp_path / "experiment", ExperimentSeed("experiment"))
    (experiment / "experiment.md").write_text(SPEC)
    _write_outputs(experiment)
    return experiment, _record(experiment)


def test_reviewed_inconclusive_answer_can_complete(reviewed) -> None:
    experiment, record = reviewed
    save_completion(experiment, record)
    assert check_completion(experiment) == []


def test_changing_a_number_file_blocks_completion(reviewed) -> None:
    experiment, record = reviewed
    save_completion(experiment, record)
    target = experiment / "outputs/evaluation.md"
    target.write_text(target.read_text() + "\n ")
    assert any("Stale evaluation" in error for error in check_completion(experiment))


@pytest.mark.parametrize("path", ["experiment.md", "outputs/report.md", "outputs/reproduce.md"])
def test_changing_a_narrative_file_only_warns(reviewed, path: str) -> None:
    experiment, record = reviewed
    save_completion(experiment, record)
    target = experiment / path
    target.write_text(target.read_text() + "\n ")
    status = assess_completion(experiment)
    assert status.errors == []
    assert any("Stale" in warning for warning in status.warnings)
    assert assess_completion(experiment, strict=True).errors


def test_a_completion_needs_only_an_outcome_and_a_conclusion(reviewed) -> None:
    experiment, _ = reviewed
    record = record_from_evidence(
        experiment, None, None, outcome="inconclusive", conclusion="Not enough data."
    )
    save_completion(experiment, record)
    assert assess_completion(experiment).label == "complete, exploratory"


def test_missing_record_is_reported(reviewed) -> None:
    experiment, _ = reviewed
    assert any("No completion record" in error for error in check_completion(experiment))


def test_review_rejection_is_enforced(reviewed) -> None:
    experiment, record = reviewed
    rejected = record.model_copy(update={"review": _write_review(experiment, "changes_requested")})
    assert "The independent review requests changes." in check_completion(experiment, rejected)


def test_stale_review_warns(reviewed) -> None:
    experiment, record = reviewed
    review = _write_review(experiment)
    (experiment / "outputs/report.md").write_text("A changed report.")
    current = record.model_copy(
        update={"report": evidence_file(experiment, "outputs/report.md"), "review": review}
    )
    status = assess_completion(experiment, current)
    assert status.errors == []
    assert "The review did not cover the current report." in status.warnings


def test_escaping_reference_is_not_followed(reviewed, tmp_path: Path) -> None:
    experiment, record = reviewed
    (tmp_path / "outside.md").write_text("outside")
    escaping = record.model_copy(
        update={"report": EvidenceFile(path="../outside.md", sha256="0" * 64)}
    )
    status = assess_completion(experiment, escaping)
    assert any("within the experiment" in warning for warning in status.warnings)


def test_supported_outcome_without_candidates_warns(reviewed) -> None:
    experiment, _ = reviewed
    status = assess_completion(experiment, _record(experiment, outcome="supported"))
    assert status.errors == []
    assert "The conclusion names no selected candidate or no baseline." in status.warnings


def test_retired_holdouts_label_the_result_exploratory(reviewed) -> None:
    experiment, _ = reviewed
    uid = "11111111-1111-1111-1111-111111111111"
    seal = HoldoutSeal(experiment_uid=uid, holdout_sha256="a" * 64, holdout_rows=1)
    write_json(experiment / "holdout_seal.json", seal.model_dump())
    record_retirement(
        LedgerEntry(
            holdout_sha256="b" * 64,
            event="retired",
            experiment_uid=uid,
            experiment_name="experiment",
            experiment_path="experiment",
            provenance_note="Old.",
            recorded_at="x",
        )
    )
    status = assess_completion(experiment, _record(experiment, outcome="supported"))
    assert status.errors == []
    assert status.label == "complete, exploratory"
    assert any("1 earlier holdouts" in reason for reason in status.exploratory)


def test_a_final_file_that_is_not_real_evidence_blocks(reviewed) -> None:
    experiment, _ = reviewed
    write_json(experiment / "outputs/final_evaluation.json", {"metrics": {"auc": 0.99}})
    forged_final = _record(
        experiment,
        outcome="supported",
        selected_candidate_id="a",
        baseline_candidate_id="b",
        evaluation=evidence_file(experiment, "outputs/final_evaluation.json"),
    )
    assert any("Invalid final evidence" in e for e in check_completion(experiment, forged_final))


def test_unknown_candidate_is_rejected(reviewed) -> None:
    experiment, _ = reviewed
    record = _record(
        experiment,
        outcome="supported",
        selected_candidate_id="a",
        baseline_candidate_id="b",
        comparison_candidate_id="c",
    )
    errors = check_completion(experiment, record)
    assert "Completion references unknown candidate: a" in errors
