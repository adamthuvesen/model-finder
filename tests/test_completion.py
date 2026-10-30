from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from agentic_ml_loop.completion import (
    CompletionRecord,
    EvidenceFile,
    ResearchReview,
    check_completion,
    evidence_file,
    save_completion,
)
from agentic_ml_loop.io import write_json
from agentic_ml_loop.runner import ExperimentSeed, init_experiment_dir
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


@pytest.mark.parametrize(
    "path", ["experiment.md", "outputs/report.md", "outputs/evaluation.md", "outputs/reproduce.md"]
)
def test_changing_reviewed_evidence_invalidates_completion(reviewed, path: str) -> None:
    experiment, record = reviewed
    save_completion(experiment, record)
    target = experiment / path
    target.write_text(target.read_text() + "\n ")
    assert any("Stale" in error for error in check_completion(experiment))


def test_missing_report_is_not_completion(reviewed) -> None:
    experiment, record = reviewed
    (experiment / "outputs/report.md").unlink()
    with pytest.raises(ValueError, match="missing"):
        save_completion(experiment, record)
    assert not (experiment / "outputs/completion.json").exists()


def test_missing_record_is_reported(reviewed) -> None:
    experiment, _ = reviewed
    assert any("Invalid completion record" in error for error in check_completion(experiment))


def test_review_rejection_is_enforced(reviewed) -> None:
    experiment, record = reviewed
    rejected = record.model_copy(update={"review": _write_review(experiment, "changes_requested")})
    assert "Independent review requests changes" in check_completion(experiment, rejected)


def test_stale_review_is_enforced(reviewed) -> None:
    experiment, record = reviewed
    review = _write_review(experiment)
    (experiment / "outputs/report.md").write_text("A changed report.")
    current = record.model_copy(
        update={"report": evidence_file(experiment, "outputs/report.md"), "review": review}
    )
    assert "Independent review did not review current report" in check_completion(
        experiment, current
    )


def test_escaping_reference_is_rejected(reviewed, tmp_path: Path) -> None:
    experiment, record = reviewed
    (tmp_path / "outside.md").write_text("outside")
    escaping = record.model_copy(
        update={"report": EvidenceFile(path="../outside.md", sha256="0" * 64)}
    )
    errors = check_completion(experiment, escaping)
    assert any("outputs/" in error for error in errors)
    assert any("within the experiment" in error for error in errors)


def test_supported_outcome_requires_candidates(reviewed) -> None:
    experiment, _ = reviewed
    record = _record(experiment, outcome="supported")
    assert any("selected candidate and baseline" in e for e in check_completion(experiment, record))


def test_retaining_a_baseline_requires_a_tested_comparison(reviewed) -> None:
    experiment, _ = reviewed
    record = _record(
        experiment,
        outcome="no_demonstrated_improvement",
        selected_candidate_id="a",
        baseline_candidate_id="a",
    )
    assert any("must be distinct" in e for e in check_completion(experiment, record))


def test_retired_holdouts_must_be_acknowledged(reviewed) -> None:
    experiment, record = reviewed
    history = experiment / "work/holdout_history/0001"
    history.mkdir(parents=True)
    (history / "consumed.json").write_text('{"holdout": {"holdout_sha256": "x"}}')
    (experiment / "holdout_seal.json").write_text("{}")
    assert any("retired 1 holdouts" in error for error in check_completion(experiment, record))
    acknowledged = record.model_copy(update={"acknowledged_prior_holdouts": 1})
    assert check_completion(experiment, acknowledged) == []


def test_supported_record_needs_real_final_evidence(reviewed) -> None:
    experiment, _ = reviewed
    forged = _record(
        experiment,
        outcome="supported",
        selected_candidate_id="a",
        baseline_candidate_id="b",
        evaluation=evidence_file(experiment, "experiment.md"),
    )
    assert "Completion evaluation must reference outputs/final_evaluation.json" in check_completion(
        experiment, forged
    )
    write_json(experiment / "outputs/final_evaluation.json", {"metrics": {"auc": 0.99}})
    forged_final = _record(
        experiment,
        outcome="supported",
        selected_candidate_id="a",
        baseline_candidate_id="b",
        evaluation=evidence_file(experiment, "outputs/final_evaluation.json"),
    )
    assert any("Invalid final evidence" in e for e in check_completion(experiment, forged_final))


def test_inconclusive_evaluation_stays_in_outputs(reviewed) -> None:
    experiment, record = reviewed
    moved = record.model_copy(update={"evaluation": evidence_file(experiment, "experiment.md")})
    assert "Completion evaluation must be under outputs/" in check_completion(experiment, moved)


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
    assert any("explicit comparison candidate" in error for error in errors)
