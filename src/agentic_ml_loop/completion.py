"""Bind a research conclusion to the evidence that was reviewed."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_ml_loop.final_evaluation import load_verified_final
from agentic_ml_loop.holdout import consumed_holdout_count
from agentic_ml_loop.io import read_text, write_json
from agentic_ml_loop.paths import (
    COMPLETION_PATH,
    FINAL_PATH,
    REVIEW_PATH,
    SPECIFICATION_FILE_NAME,
)
from agentic_ml_loop.results import RESULTS_FILE_NAME, RecordedResult, load_results
from agentic_ml_loop.run_evidence import validate_run_evidence
from agentic_ml_loop.splits import Sha256
from agentic_ml_loop.validation import validate_experiment


class EvidenceFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: Sha256


class CompletionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    outcome: Literal["supported", "no_demonstrated_improvement", "inconclusive"]
    conclusion: str = Field(min_length=1)
    limitations: list[str] = Field(min_length=1)
    selected_candidate_id: str | None = None
    baseline_candidate_id: str | None = None
    comparison_candidate_id: str | None = None
    acknowledged_prior_holdouts: int = 0
    acknowledged_overlap: str | None = None
    specification: EvidenceFile
    results: EvidenceFile
    report: EvidenceFile
    evaluation: EvidenceFile
    review: EvidenceFile
    reproduction: EvidenceFile


class ResearchReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    reviewer: str = Field(min_length=1)
    verdict: Literal["accepted", "changes_requested"]
    specification_sha256: Sha256
    results_sha256: Sha256
    report_sha256: Sha256
    evaluation_sha256: Sha256
    findings: list[str]
    limitations: list[str] = Field(min_length=1)


def _confined_file(experiment_dir: Path, relative_path: str) -> Path:
    root = experiment_dir.resolve()
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Evidence path must stay within the experiment: {relative_path}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Evidence file missing or outside experiment: {relative_path}")
    return resolved


def evidence_file(experiment_dir: Path, relative_path: str) -> EvidenceFile:
    content = _confined_file(experiment_dir, relative_path).read_bytes()
    if not content.strip():
        raise ValueError(f"Evidence file is empty: {relative_path}")
    return EvidenceFile(path=relative_path, sha256=hashlib.sha256(content).hexdigest())


def _load_record(experiment_dir: Path) -> CompletionRecord:
    path = _confined_file(experiment_dir, COMPLETION_PATH)
    return CompletionRecord.model_validate_json(read_text(path))


def _reference_errors(experiment_dir: Path, record: CompletionRecord) -> list[str]:
    references = {
        "specification": record.specification,
        "results": record.results,
        "report": record.report,
        "evaluation": record.evaluation,
        "review": record.review,
        "reproduction": record.reproduction,
    }
    errors: list[str] = []
    if record.specification.path != SPECIFICATION_FILE_NAME:
        errors.append(f"Completion specification must reference {SPECIFICATION_FILE_NAME}")
    if record.results.path != RESULTS_FILE_NAME:
        errors.append(f"Completion results must reference {RESULTS_FILE_NAME}")
    if not record.report.path.startswith("outputs/"):
        errors.append("Completion report must be under outputs/")
    for name, reference in references.items():
        try:
            if evidence_file(experiment_dir, reference.path).sha256 != reference.sha256:
                errors.append(f"Stale {name} evidence: {reference.path}")
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def _review_errors(experiment_dir: Path, record: CompletionRecord) -> list[str]:
    try:
        review = ResearchReview.model_validate_json(
            read_text(_confined_file(experiment_dir, record.review.path))
        )
    except (OSError, ValueError, ValidationError) as exc:
        return [f"Invalid independent review: {exc}"]
    errors: list[str] = []
    if review.verdict != "accepted":
        errors.append("Independent review requests changes")
    for name, reviewed, current in (
        ("specification", review.specification_sha256, record.specification.sha256),
        ("results", review.results_sha256, record.results.sha256),
        ("report", review.report_sha256, record.report.sha256),
        ("evaluation", review.evaluation_sha256, record.evaluation.sha256),
    ):
        if reviewed != current:
            errors.append(f"Independent review did not review current {name}")
    return errors


def _comparison_errors(record: CompletionRecord) -> list[str]:
    if record.outcome == "inconclusive":
        return []
    errors: list[str] = []
    selected, baseline, comparison = (
        record.selected_candidate_id,
        record.baseline_candidate_id,
        record.comparison_candidate_id,
    )
    retains_baseline = record.outcome == "no_demonstrated_improvement" and selected == baseline
    if not selected or not baseline:
        errors.append("A supported comparison requires a selected candidate and baseline")
    elif selected == baseline and not (retains_baseline and comparison and comparison != baseline):
        errors.append(
            "Selected candidate and baseline must be distinct, "
            "or retaining the baseline requires a distinct tested comparison candidate"
        )
    if comparison is not None and not retains_baseline:
        errors.append(
            "An explicit comparison candidate is only valid when retaining the baseline "
            "after no demonstrated improvement"
        )
    return errors


def _final_evidence_errors(experiment_dir: Path, record: CompletionRecord) -> list[str]:
    """Bind a claim about a model to the real freeze and final evaluation."""
    if record.outcome == "inconclusive":
        if not record.evaluation.path.startswith("outputs/"):
            return ["Completion evaluation must be under outputs/"]
        return []
    if record.evaluation.path != FINAL_PATH:
        return [f"Completion evaluation must reference {FINAL_PATH}"]
    try:
        selection, final = load_verified_final(experiment_dir)
    except (OSError, ValueError) as exc:
        return [f"Invalid final evidence: {exc}"]
    errors: list[str] = []
    frozen = selection.selected_result.candidate_id
    if record.selected_candidate_id != frozen:
        errors.append(f"Completion selects {record.selected_candidate_id}, but {frozen} was frozen")
    if final.overlap.needs_acknowledgement and not (
        record.acknowledged_overlap and record.acknowledged_overlap.strip()
    ):
        errors.append(
            f"The final evaluation reports shared or incomparable rows: {final.overlap.model_dump()}. "
            "Say why that is acceptable in acknowledged_overlap."
        )
    return errors


def _candidate_errors(
    experiment_dir: Path, record: CompletionRecord, results: list[RecordedResult]
) -> list[str]:
    errors = [
        error for result in results for error in validate_run_evidence(experiment_dir, result)
    ]
    by_id = {result.candidate_id: result for result in results}
    for candidate_id in (
        record.selected_candidate_id,
        record.baseline_candidate_id,
        record.comparison_candidate_id,
    ):
        if candidate_id is None:
            continue
        entry = by_id.get(candidate_id)
        if entry is None:
            errors.append(f"Completion references unknown candidate: {candidate_id}")
        elif entry.comparison_context.spec_sha256 != record.specification.sha256:
            errors.append(f"Candidate used a different specification: {candidate_id}")
    return errors


def check_completion(experiment_dir: Path, record: CompletionRecord | None = None) -> list[str]:
    errors = [
        message
        for message in validate_experiment(experiment_dir)
        if not message.startswith("warning:")
    ]
    if errors:
        return errors
    if record is None:
        try:
            record = _load_record(experiment_dir)
        except (OSError, ValueError, ValidationError) as exc:
            return [f"Invalid completion record: {exc}"]
    errors = _reference_errors(experiment_dir, record)
    if errors:
        return errors
    retired = consumed_holdout_count(experiment_dir)
    if record.acknowledged_prior_holdouts != retired:
        errors.append(
            f"This experiment retired {retired} holdouts. Earlier estimates are exploratory. "
            "Set acknowledged_prior_holdouts to that count and say so in the limitations."
        )
    return [
        *errors,
        *_review_errors(experiment_dir, record),
        *_comparison_errors(record),
        *_final_evidence_errors(experiment_dir, record),
        *_candidate_errors(experiment_dir, record, load_results(experiment_dir)),
    ]


def save_completion(experiment_dir: Path, record: CompletionRecord) -> Path:
    """Save a record only after its evidence passes the completion check."""
    errors = check_completion(experiment_dir, record)
    if errors:
        raise ValueError("; ".join(errors))
    path = experiment_dir / COMPLETION_PATH
    write_json(path, record.model_dump())
    return path


def write_review(
    experiment_dir: Path,
    reviewer: str,
    verdict: Literal["accepted", "changes_requested"],
    report_path: str,
    evaluation_path: str,
    findings: list[str],
    limitations: list[str],
) -> Path:
    """Bind a reviewer's verdict to the current hashes of what was reviewed."""
    review = ResearchReview(
        reviewer=reviewer,
        verdict=verdict,
        specification_sha256=evidence_file(experiment_dir, SPECIFICATION_FILE_NAME).sha256,
        results_sha256=evidence_file(experiment_dir, RESULTS_FILE_NAME).sha256,
        report_sha256=evidence_file(experiment_dir, report_path).sha256,
        evaluation_sha256=evidence_file(experiment_dir, evaluation_path).sha256,
        findings=findings,
        limitations=limitations,
    )
    path = experiment_dir / REVIEW_PATH
    write_json(path, review.model_dump())
    return path


def record_from_evidence(
    experiment_dir: Path,
    report_path: str,
    evaluation_path: str,
    reproduction_path: str,
    **claim: object,
) -> CompletionRecord:
    """Build a record whose references carry the current hash of each file."""
    return CompletionRecord.model_validate(
        {
            **claim,
            "specification": evidence_file(experiment_dir, SPECIFICATION_FILE_NAME),
            "results": evidence_file(experiment_dir, RESULTS_FILE_NAME),
            "report": evidence_file(experiment_dir, report_path),
            "evaluation": evidence_file(experiment_dir, evaluation_path),
            "review": evidence_file(experiment_dir, REVIEW_PATH),
            "reproduction": evidence_file(experiment_dir, reproduction_path),
        }
    )
