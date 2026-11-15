"""Label a research conclusion by the evidence behind it."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model_finder.final_evaluation import load_verified_final
from model_finder.holdout import holdout_read_labels, is_sealed, read_holdout_seal
from model_finder.io import read_text, write_json
from model_finder.labels import assess_final, evidence_labels
from model_finder.paths import (
    COMPLETION_PATH,
    FINAL_PATH,
    REVIEW_PATH,
    SPECIFICATION_FILE_NAME,
)
from model_finder.results import RESULTS_FILE_NAME, RecordedResult, load_results
from model_finder.run_evidence import run_evidence_staleness, validate_run_evidence
from model_finder.splits import Sha256
from model_finder.validation import NO_HOLDOUT_WARNING, validate_experiment


class EvidenceFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: Sha256


class CompletionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    outcome: Literal["supported", "no_demonstrated_improvement", "inconclusive"]
    conclusion: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list[str])
    selected_candidate_id: str | None = None
    baseline_candidate_id: str | None = None
    comparison_candidate_id: str | None = None
    acknowledged_overlap: str | None = None
    specification: EvidenceFile
    results: EvidenceFile
    report: EvidenceFile | None = None
    evaluation: EvidenceFile | None = None
    review: EvidenceFile | None = None
    reproduction: EvidenceFile | None = None


class ResearchReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    reviewer: str = Field(min_length=1)
    verdict: Literal["accepted", "changes_requested"]
    specification_sha256: Sha256
    results_sha256: Sha256
    report_sha256: Sha256 | None = None
    evaluation_sha256: Sha256 | None = None
    findings: list[str] = Field(default_factory=list[str])
    limitations: list[str] = Field(default_factory=list[str])


def _confined_file(experiment_dir: Path, relative_path: str) -> Path:
    root = experiment_dir.resolve()
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Evidence path must stay within the experiment: {relative_path}. "
            "Fix: use a path relative to the experiment directory"
        )
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(
            f"{relative_path}: Evidence file missing or outside experiment. "
            "Fix: create the file, or correct the path"
        )
    return resolved


def evidence_file(experiment_dir: Path, relative_path: str) -> EvidenceFile:
    content = _confined_file(experiment_dir, relative_path).read_bytes()
    if not content.strip():
        raise ValueError(f"{relative_path}: Evidence file is empty. Fix: write its content")
    return EvidenceFile(path=relative_path, sha256=hashlib.sha256(content).hexdigest())


def _load_record(experiment_dir: Path) -> CompletionRecord:
    path = _confined_file(experiment_dir, COMPLETION_PATH)
    return CompletionRecord.model_validate_json(read_text(path))


@dataclass
class CompletionStatus:
    """One result for a person to read. Errors block. Exploratory reasons and warnings only label."""

    errors: list[str] = field(default_factory=list[str])
    exploratory: list[str] = field(default_factory=list[str])
    warnings: list[str] = field(default_factory=list[str])

    @property
    def label(self) -> str:
        if self.errors:
            return "not complete"
        return "complete, exploratory" if self.exploratory else "complete"


def _check_references(
    experiment_dir: Path, record: CompletionRecord, status: CompletionStatus
) -> None:
    """A changed number file blocks. A changed narrative file warns."""
    for name, reference, blocks in (
        ("results", record.results, True),
        ("evaluation", record.evaluation, True),
        ("specification", record.specification, False),
        ("report", record.report, False),
        ("review", record.review, False),
        ("reproduction", record.reproduction, False),
    ):
        if reference is None:
            continue
        try:
            if evidence_file(experiment_dir, reference.path).sha256 == reference.sha256:
                continue
            problem = f"Stale {name} evidence: {reference.path}. Run record-completion again."
        except ValueError as exc:
            problem = str(exc)
        (status.errors if blocks else status.warnings).append(problem)


def _check_review(experiment_dir: Path, record: CompletionRecord, status: CompletionStatus) -> None:
    if record.review is None:
        status.warnings.append("No independent review was recorded.")
        return
    try:
        review = ResearchReview.model_validate_json(
            read_text(_confined_file(experiment_dir, record.review.path))
        )
    except (OSError, ValueError, ValidationError) as exc:
        status.warnings.append(f"The review cannot be read: {exc}")
        return
    if review.verdict != "accepted":
        status.errors.append("The independent review requests changes.")
    for name, reviewed, current in (
        ("specification", review.specification_sha256, record.specification),
        ("results", review.results_sha256, record.results),
        ("report", review.report_sha256, record.report),
        ("evaluation", review.evaluation_sha256, record.evaluation),
    ):
        if reviewed is not None and current is not None and reviewed != current.sha256:
            status.warnings.append(f"The review did not cover the current {name}.")


def _comparison_warnings(record: CompletionRecord) -> list[str]:
    if record.outcome == "inconclusive":
        return []
    selected, baseline = record.selected_candidate_id, record.baseline_candidate_id
    if not selected or not baseline:
        return ["The conclusion names no selected candidate or no baseline."]
    if selected == baseline and not record.comparison_candidate_id:
        return ["The selected candidate is the baseline, and no tested challenger is named."]
    return []


def _check_final(experiment_dir: Path, record: CompletionRecord, status: CompletionStatus) -> None:
    """Forged or edited evidence blocks. A missing or reused holdout only labels the result."""
    if (experiment_dir / FINAL_PATH).is_file():
        try:
            selection, final = load_verified_final(experiment_dir)
        except (OSError, ValueError) as exc:
            status.errors.append(f"Invalid final evidence: {exc}")
            return
        frozen = selection.selected_result.candidate_id
        if record.selected_candidate_id not in (None, frozen):
            status.errors.append(
                f"Completion selects {record.selected_candidate_id}, but {frozen} was frozen"
            )
        assessment = assess_final(experiment_dir, final)
        status.exploratory.extend(assessment.exploratory)
        status.warnings.extend(assessment.warnings)
        if final.overlap.is_large and not (record.acknowledged_overlap or "").strip():
            status.errors.append(
                "A large share of holdout rows also appears in search rows. "
                "Say why that is acceptable with --acknowledged-overlap <text>."
            )
        return
    if not is_sealed(experiment_dir):
        status.exploratory.append("No holdout was sealed. Scores come from validation rows.")
        return
    status.exploratory.append("A holdout is sealed, but no final evaluation is on record.")
    status.exploratory.extend(
        holdout_read_labels(experiment_dir, read_holdout_seal(experiment_dir))
    )


def _check_candidates(
    experiment_dir: Path,
    record: CompletionRecord,
    results: list[RecordedResult],
    status: CompletionStatus,
) -> list[str]:
    """Return what changed since the runs, which warns unless a strict check was asked for."""
    stale: list[str] = []
    for result in results:
        problems = validate_run_evidence(experiment_dir, result)
        status.errors.extend(problems)
        if not problems:
            stale.extend(run_evidence_staleness(experiment_dir, result))
    known = {result.candidate_id for result in results}
    status.errors.extend(
        f"Completion references unknown candidate: {candidate_id}"
        for candidate_id in (
            record.selected_candidate_id,
            record.baseline_candidate_id,
            record.comparison_candidate_id,
        )
        if candidate_id is not None and candidate_id not in known
    )
    return stale


def assess_completion(
    experiment_dir: Path, record: CompletionRecord | None = None, *, strict: bool = False
) -> CompletionStatus:
    """Give one labelled result. ``strict`` turns every label and warning into a failure."""
    status = CompletionStatus()
    for message in validate_experiment(experiment_dir):
        if message == NO_HOLDOUT_WARNING:
            continue
        if message.startswith("warning: "):
            status.warnings.append(message.removeprefix("warning: "))
        else:
            status.errors.append(message)
    if status.errors:
        return status
    if record is None:
        if not (experiment_dir / COMPLETION_PATH).is_file():
            status.errors.append(
                "No completion record. Run `experiment record-completion --experiment "
                f"{experiment_dir} --outcome <outcome> --conclusion <text>`."
            )
            return status
        try:
            record = _load_record(experiment_dir)
        except (OSError, ValueError, ValidationError) as exc:
            status.errors.append(f"Invalid completion record: {exc}")
            return status
    _check_references(experiment_dir, record, status)
    if status.errors:
        return status
    _check_review(experiment_dir, record, status)
    status.warnings.extend(_comparison_warnings(record))
    status.exploratory.extend(evidence_labels(experiment_dir))
    _check_final(experiment_dir, record, status)
    stale = _check_candidates(experiment_dir, record, load_results(experiment_dir), status)
    if strict:
        status.errors.extend([*stale, *status.exploratory, *status.warnings])
        status.exploratory, status.warnings = [], []
    else:
        status.warnings.extend(stale)
    return status


def check_completion(
    experiment_dir: Path, record: CompletionRecord | None = None, *, strict: bool = False
) -> list[str]:
    return assess_completion(experiment_dir, record, strict=strict).errors


def save_completion(experiment_dir: Path, record: CompletionRecord) -> Path:
    """Save a record unless something blocks it."""
    errors = check_completion(experiment_dir, record)
    if errors:
        raise ValueError("; ".join(errors))
    path = experiment_dir / COMPLETION_PATH
    write_json(path, record.model_dump())
    return path


def _optional_evidence(experiment_dir: Path, relative_path: str | None) -> EvidenceFile | None:
    return evidence_file(experiment_dir, relative_path) if relative_path else None


def _final_if_present(experiment_dir: Path) -> str | None:
    return FINAL_PATH if (experiment_dir / FINAL_PATH).is_file() else None


def write_review(
    experiment_dir: Path,
    reviewer: str,
    verdict: Literal["accepted", "changes_requested"],
    report_path: str | None,
    findings: list[str],
    limitations: list[str],
) -> Path:
    """Bind a reviewer's verdict to the current hashes of what was reviewed."""
    report = _optional_evidence(experiment_dir, report_path)
    evaluation = _optional_evidence(experiment_dir, _final_if_present(experiment_dir))
    review = ResearchReview(
        reviewer=reviewer,
        verdict=verdict,
        specification_sha256=evidence_file(experiment_dir, SPECIFICATION_FILE_NAME).sha256,
        results_sha256=evidence_file(experiment_dir, RESULTS_FILE_NAME).sha256,
        report_sha256=report.sha256 if report else None,
        evaluation_sha256=evaluation.sha256 if evaluation else None,
        findings=findings,
        limitations=limitations,
    )
    path = experiment_dir / REVIEW_PATH
    write_json(path, review.model_dump())
    return path


def record_from_evidence(
    experiment_dir: Path,
    report_path: str | None,
    reproduction_path: str | None,
    **claim: object,
) -> CompletionRecord:
    """Build a record whose references carry the current hash of each file that exists."""
    review_path = REVIEW_PATH if (experiment_dir / REVIEW_PATH).is_file() else None
    return CompletionRecord.model_validate(
        {
            **claim,
            "specification": evidence_file(experiment_dir, SPECIFICATION_FILE_NAME),
            "results": evidence_file(experiment_dir, RESULTS_FILE_NAME),
            "report": _optional_evidence(experiment_dir, report_path),
            "evaluation": _optional_evidence(experiment_dir, _final_if_present(experiment_dir)),
            "review": _optional_evidence(experiment_dir, review_path),
            "reproduction": _optional_evidence(experiment_dir, reproduction_path),
        }
    )
