"""Freeze a selection, then evaluate it once on the sealed holdout."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict

from agentic_ml_loop.candidate_result import (
    CandidateResult,
    CandidateRunner,
    MetricValues,
    SearchMetrics,
)
from agentic_ml_loop.data_audit import (
    DataAudit,
    OverlapReport,
    audit_search_splits,
    overlap_report,
    target_shift_concern,
)
from agentic_ml_loop.holdout import (
    HoldoutSeal,
    check_replacement_rows,
    commit_seal,
    consumed_holdout_count,
    history_seals,
    load_sealed_holdout,
    load_search_splits,
    place_staged_file,
    read_holdout_seal,
    remove_other_sealed_files,
    replace_read_evidence,
    stage_replacement_rows,
    verify_sealed_holdout,
    withdraw_selection,
)
from agentic_ml_loop.io import file_sha256, json_text, read_text, utc_now, write_json
from agentic_ml_loop.ledger import (
    LedgerEntry,
    ProvenanceNote,
    claim_for_evaluation,
    entries_for,
    record_final,
    record_retirement,
)
from agentic_ml_loop.paths import (
    ATTEMPT_PATH,
    DATA_DIRNAME,
    FINAL_PATH,
    HOLDOUT_HISTORY_DIR,
    HOLDOUT_SEAL_FILE_NAME,
    REBUILDS_PATH,
    SELECTION_PATH,
    repository_relative_path,
)
from agentic_ml_loop.results import (
    RecordedResult,
    comparison_errors,
    load_results,
    results_lock,
    results_path,
    results_sha256,
    results_text_sha256,
    write_results,
)
from agentic_ml_loop.run_evidence import (
    CodeFingerprint,
    RunEvidence,
    latest_recorded_results,
    run_evidence_staleness,
    validate_run_evidence,
)
from agentic_ml_loop.sources.manifest import nondeterministic_pull_label
from agentic_ml_loop.spec import get_key_columns
from agentic_ml_loop.splits import Sha256, search_data_identity

_FROZEN_FIELDS = (
    "candidate_id",
    "model_family",
    "feature_set",
    "split_strategy",
    "objective_metric",
    "objective_direction",
    "objective_score",
    "selected_features",
    "random_seed",
    "train_validation_gap",
    "metrics",
    "hyperparameters",
)


class FrozenSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    selected_result: RecordedResult
    results_sha256: Sha256
    code_fingerprint: CodeFingerprint
    data_audit: DataAudit


class FinalAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    selected_run_id: str
    selection_sha256: Sha256
    holdout: HoldoutSeal
    started_at: str


class FinalEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    selected_run_id: str
    selection_sha256: Sha256
    attempt_sha256: Sha256
    holdout: HoldoutSeal
    completed_at: str
    metrics: MetricValues
    selection_metrics: SearchMetrics
    prior_holdouts: int
    exploratory_reasons: list[str]
    holdout_target_concern: str | None
    overlap: OverlapReport
    limitations: list[str]


class ConsumedHoldout(BaseModel):
    """The permanent record that an earlier holdout no longer gives an unbiased estimate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    holdout: HoldoutSeal
    provenance_note: ProvenanceNote
    exploratory_run_ids: list[str]
    resealed_at: str


def _raise_errors(messages: list[str]) -> None:
    if messages:
        raise ValueError("; ".join(messages))


def _recorded_fingerprint(experiment_dir: Path, selected: RecordedResult) -> CodeFingerprint:
    evidence = RunEvidence.model_validate_json(read_text(experiment_dir / selected.run_evidence))
    return evidence.code_fingerprint


def freeze_candidate(
    experiment_dir: Path,
    candidate_id: str,
    target_column: str,
    *,
    reread_reason: str | None = None,
) -> FrozenSelection:
    """Bind one candidate and report split drift. A later freeze replaces an unread selection."""
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    with results_lock(experiment_dir):
        results = load_results(experiment_dir)
        _raise_errors(comparison_errors(results))
        matches = [result for result in results if result.candidate_id == candidate_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one candidate named {candidate_id}")
        selected = matches[0]
        _raise_errors(validate_run_evidence(experiment_dir, selected))
        audit = audit_search_splits(
            load_search_splits(experiment_dir), target_column, get_key_columns(experiment_dir)
        )
        selection = FrozenSelection(
            selected_result=selected,
            results_sha256=results_sha256(experiment_dir),
            code_fingerprint=_recorded_fingerprint(experiment_dir, selected),
            data_audit=audit,
        )
        path = experiment_dir / SELECTION_PATH
        if path.exists() and FrozenSelection.model_validate_json(read_text(path)) == selection:
            return selection
        withdraw_selection(experiment_dir, reread_reason)
        write_json(path, selection.model_dump(mode="json"))
        return selection


def _load_selection(experiment_dir: Path) -> FrozenSelection:
    path = experiment_dir / SELECTION_PATH
    if not path.is_file():
        raise ValueError("No frozen selection. Run freeze-candidate before evaluate-final.")
    selection = FrozenSelection.model_validate_json(read_text(path))
    _raise_errors(validate_run_evidence(experiment_dir, selection.selected_result))
    if selection.results_sha256 != results_sha256(experiment_dir):
        raise ValueError("Results changed after selection was frozen")
    if search_data_identity(load_search_splits(experiment_dir)) != (
        selection.selected_result.comparison_context.data
    ):
        raise ValueError("Search data changed after selection was frozen")
    return selection


def _require_ledger_chain(
    experiment_dir: Path, final: FinalEvaluation, attempt_sha256: str
) -> None:
    """The engine records a read and its final file's hash. Other evidence was not made by it."""
    if final.attempt_sha256 != attempt_sha256:
        raise ValueError("Final evidence does not chain to the attempt file")
    entries = [
        entry
        for entry in entries_for(final.holdout.holdout_sha256)
        if entry.experiment_uid == final.holdout.experiment_uid
    ]
    if not entries:
        return
    if not any(
        entry.event == "evaluated" and entry.attempt_sha256 == attempt_sha256 for entry in entries
    ):
        raise ValueError("The ledger has no evaluation of this holdout that matches the attempt")
    final_sha256 = file_sha256(experiment_dir / FINAL_PATH)
    finalized = [
        entry
        for entry in entries
        if entry.event == "finalized"
        and entry.attempt_sha256 == attempt_sha256
        and entry.final_sha256 == final_sha256
    ]
    if not finalized:
        raise ValueError("The ledger has no record of this final file. It was edited or forged.")


def load_verified_final(experiment_dir: Path) -> tuple[FrozenSelection, FinalEvaluation]:
    """Return final evidence only when it matches its frozen selection, attempt, and seal."""
    selection = _load_selection(experiment_dir)
    attempt_path = experiment_dir / ATTEMPT_PATH
    if not attempt_path.is_file() or not (experiment_dir / FINAL_PATH).is_file():
        raise ValueError("No completed final evaluation")
    attempt = FinalAttempt.model_validate_json(read_text(attempt_path))
    if attempt.selection_sha256 != file_sha256(experiment_dir / SELECTION_PATH):
        raise ValueError("Final attempt does not match the frozen selection")
    if attempt.holdout != read_holdout_seal(experiment_dir):
        raise ValueError("Final attempt does not match the current seal")
    return selection, _completed_evaluation(experiment_dir, selection, attempt)


def _completed_evaluation(
    experiment_dir: Path, selection: FrozenSelection, attempt: FinalAttempt
) -> FinalEvaluation:
    final = FinalEvaluation.model_validate_json(read_text(experiment_dir / FINAL_PATH))
    if (
        final.selected_run_id != attempt.selected_run_id
        or final.selection_sha256 != attempt.selection_sha256
        or final.attempt_sha256 != file_sha256(experiment_dir / ATTEMPT_PATH)
        or final.holdout != attempt.holdout
        or final.selection_metrics != selection.selected_result.metrics
    ):
        raise ValueError("Final evidence does not match the frozen selection or its attempt")
    _require_ledger_chain(experiment_dir, final, file_sha256(experiment_dir / ATTEMPT_PATH))
    return final


@dataclass(frozen=True)
class FinalAssessment:
    """Why an estimate is exploratory, and what a reader should know."""

    exploratory: list[str]
    warnings: list[str]


def holdout_read_labels(experiment_dir: Path, seal: HoldoutSeal) -> list[str]:
    """Label search that continued after a read, from the ledger. Workspace edits cannot lower it."""
    reasons = [
        f"Search continued after reading the holdout: {entry.provenance_note}"
        for entry in entries_for(seal.holdout_sha256)
        if entry.event == "reopened" and entry.experiment_uid == seal.experiment_uid
    ]
    retired = consumed_holdout_count(experiment_dir)
    if retired:
        reasons.append(
            f"{retired} earlier holdouts were read, so selection may be informed by them."
        )
    return reasons


def evidence_labels(experiment_dir: Path) -> list[str]:
    """Label overrides that touched the rows or the leaderboard, whatever the claim."""
    pull = nondeterministic_pull_label(experiment_dir / DATA_DIRNAME)
    return [*([pull] if pull else []), *rebuild_labels(experiment_dir)]


def leaderboard_label(experiment_dir: Path) -> str | None:
    """Return one short line when the leaderboard's scores are exploratory."""
    if not (experiment_dir / HOLDOUT_SEAL_FILE_NAME).is_file():
        return "Exploratory: no holdout is sealed."
    if evidence_labels(experiment_dir):
        return "Exploratory: an override changed this work's data or results."
    seal = read_holdout_seal(experiment_dir)
    read_again = any(
        entry.event == "evaluated"
        and entry.provenance_note is not None
        and entry.experiment_uid == seal.experiment_uid
        for entry in entries_for(seal.holdout_sha256)
    )
    if read_again or holdout_read_labels(experiment_dir, seal):
        return "Exploratory: a holdout read informed or repeated this work."
    return None


def assess_final(experiment_dir: Path, final: FinalEvaluation) -> FinalAssessment:
    """Label a verified final. Nothing here refuses it."""
    exploratory = list(
        dict.fromkeys(
            [*final.exploratory_reasons, *holdout_read_labels(experiment_dir, final.holdout)]
        )
    )
    warnings: list[str] = []
    entries = [
        entry
        for entry in entries_for(final.holdout.holdout_sha256)
        if entry.experiment_uid == final.holdout.experiment_uid
    ]
    if not entries:
        exploratory.append(
            "This machine's ledger has no record of this read, so the estimate is unverified."
        )
    elif not any(
        entry.experiment_path == repository_relative_path(experiment_dir) for entry in entries
    ):
        warnings.append(f"This evidence was produced in {entries[0].experiment_path}, not here.")
    if final.overlap.is_large:
        exploratory.append(
            f"A large share of holdout rows also appears in search rows: {final.overlap.model_dump()}"
        )
    elif final.overlap.needs_warning:
        warnings.append(
            f"Splits share rows, or a comparison could not run: {final.overlap.model_dump()}"
        )
    return FinalAssessment(exploratory=exploratory, warnings=warnings)


def selection_staleness(experiment_dir: Path, selection: FrozenSelection) -> list[str]:
    return run_evidence_staleness(experiment_dir, selection.selected_result)


def _check_rerun_matches(selected: RecordedResult, rerun: CandidateResult) -> None:
    for name in _FROZEN_FIELDS:
        if getattr(rerun, name) != getattr(selected, name):
            raise ValueError(f"Final rerun changed frozen selection field: {name}")


def evaluate_final(
    experiment_dir: Path,
    candidate_runners: Mapping[str, CandidateRunner],
    *,
    reread_reason: str | None = None,
) -> FinalEvaluation:
    """Check the frozen evidence and the seal hash, then score the sealed rows.

    One read per seal is the default. A reason allows another read, recorded in the ledger and
    labelled exploratory.
    """
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    with results_lock(experiment_dir):
        selection = _load_selection(experiment_dir)
        selected = selection.selected_result
        selection_sha256 = file_sha256(experiment_dir / SELECTION_PATH)
        seal = read_holdout_seal(experiment_dir)
        attempt_path = experiment_dir / ATTEMPT_PATH
        final_path = experiment_dir / FINAL_PATH

        if attempt_path.exists() and reread_reason is not None:
            replace_read_evidence(experiment_dir)
        if attempt_path.exists():
            attempt = FinalAttempt.model_validate_json(read_text(attempt_path))
            if attempt.selection_sha256 != selection_sha256 or attempt.holdout != seal:
                raise ValueError("Final attempt does not match the frozen selection or the seal")
            if not final_path.exists():
                raise ValueError(
                    "Final evaluation was attempted but did not finish. Retry with "
                    "`--allow-reread --reason <text>`, which labels the estimate exploratory."
                )
            return _completed_evaluation(experiment_dir, selection, attempt)
        if final_path.exists():
            raise ValueError("Final evidence has no recorded evaluation attempt")
        if selected.candidate_id not in candidate_runners:
            raise ValueError(f"Frozen candidate is not available: {selected.candidate_id}")

        verify_sealed_holdout(seal)
        search = load_search_splits(experiment_dir)
        if search_data_identity(search) != selected.comparison_context.data:
            raise ValueError("Search data changed after selection was frozen")

        fitted = candidate_runners[selected.candidate_id](search)
        _check_rerun_matches(selected, fitted.result)

        attempt = FinalAttempt(
            selected_run_id=selected.run_id,
            selection_sha256=selection_sha256,
            holdout=seal,
            started_at=utc_now(),
        )
        attempt_payload = attempt.model_dump(mode="json")
        attempt_sha256 = hashlib.sha256(json_text(attempt_payload).encode()).hexdigest()
        earlier_reads = entries_for(seal.holdout_sha256)
        claim_for_evaluation(
            LedgerEntry(
                holdout_sha256=seal.holdout_sha256,
                event="evaluated",
                experiment_uid=seal.experiment_uid,
                experiment_name=experiment_dir.resolve().name,
                experiment_path=repository_relative_path(experiment_dir),
                attempt_sha256=attempt_sha256,
                provenance_note=reread_reason if earlier_reads else None,
                recorded_at=utc_now(),
            )
        )
        write_json(attempt_path, attempt_payload)

        holdout = load_sealed_holdout(seal)
        metrics = fitted.score_rows(holdout.holdout)
        target = selection.data_audit.target_column

        if (
            file_sha256(experiment_dir / SELECTION_PATH) != selection_sha256
            or results_sha256(experiment_dir) != selection.results_sha256
            or file_sha256(attempt_path) != attempt_sha256
        ):
            raise ValueError("Frozen evidence changed during final evaluation")
        final = FinalEvaluation(
            selected_run_id=selected.run_id,
            selection_sha256=selection_sha256,
            attempt_sha256=attempt_sha256,
            holdout=seal,
            completed_at=utc_now(),
            metrics=metrics,
            selection_metrics=selected.metrics,
            prior_holdouts=consumed_holdout_count(experiment_dir),
            exploratory_reasons=[
                *(
                    [f"The holdout file was read before this estimate: {reread_reason}"]
                    if earlier_reads
                    else []
                ),
                *holdout_read_labels(experiment_dir, seal),
            ],
            holdout_target_concern=target_shift_concern(
                pd.to_numeric(search.train[target], errors="coerce").dropna(),
                pd.to_numeric(holdout.holdout[target], errors="coerce").dropna(),
                "holdout",
            ),
            overlap=overlap_report(search, holdout, get_key_columns(experiment_dir)),
            limitations=[
                "Final evaluation refits the frozen implementation. "
                "It is not a serialized model replay.",
                "The seal stops honest and shortest-path agents. It does not stop a deliberate one.",
                *selection_staleness(experiment_dir, selection),
            ],
        )
        final_payload = final.model_dump(mode="json")
        record_final(
            LedgerEntry(
                holdout_sha256=seal.holdout_sha256,
                event="finalized",
                experiment_uid=seal.experiment_uid,
                experiment_name=experiment_dir.resolve().name,
                experiment_path=repository_relative_path(experiment_dir),
                attempt_sha256=attempt_sha256,
                final_sha256=hashlib.sha256(json_text(final_payload).encode()).hexdigest(),
                recorded_at=utc_now(),
            )
        )
        write_json(final_path, final_payload)
        return final


def _write_history(experiment_dir: Path, generation: int, record: ConsumedHoldout) -> None:
    """Commit the history with one rename, so a crash leaves all of it or none of it."""
    history_root = experiment_dir / HOLDOUT_HISTORY_DIR
    pending = history_root / f".pending-{generation:04d}"
    shutil.rmtree(pending, ignore_errors=True)
    pending.mkdir(parents=True)
    for relative in (SELECTION_PATH, ATTEMPT_PATH, FINAL_PATH):
        evidence = experiment_dir / relative
        if evidence.exists():
            shutil.copy2(evidence, pending / evidence.name)
    write_json(pending / "consumed.json", record.model_dump(mode="json"))
    pending.replace(history_root / f"{generation:04d}")


def _run_label(result: RecordedResult) -> str:
    return f"{result.candidate_id} ({result.run_id})"


def _describe_changes(previous: list[RecordedResult], rebuilt: list[RecordedResult]) -> list[str]:
    before = {result.run_id: result for result in previous}
    after = {result.run_id: result for result in rebuilt}
    changes = [f"Added {_run_label(after[run])}." for run in sorted(after.keys() - before.keys())]
    changes += [
        f"Removed {_run_label(before[run])}." for run in sorted(before.keys() - after.keys())
    ]
    for run in sorted(before.keys() & after.keys()):
        old, new = before[run].model_dump(mode="json"), after[run].model_dump(mode="json")
        changes += [
            f"Changed {_run_label(after[run])} {name}: {old[name]} -> {new[name]}."
            for name in old
            if old[name] != new[name]
        ]
    return changes


def rebuild_labels(experiment_dir: Path) -> list[str]:
    path = experiment_dir / REBUILDS_PATH
    if not path.is_file():
        return []
    return [
        f"A rebuild changed recorded results: {reason}" for reason in json.loads(read_text(path))
    ]


def rebuild_results(experiment_dir: Path, *, change_reason: str | None = None) -> str:
    """Repair a damaged leaderboard from run records, and say what changed.

    A readable leaderboard that disagrees with the records needs a recorded reason. A run record
    is a plain file, so an edit to it and to the leaderboard together goes unseen.
    """
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    with results_lock(experiment_dir):
        rebuilt = latest_recorded_results(experiment_dir)
        rebuilt_sha256 = results_text_sha256(rebuilt)
        if (
            results_path(experiment_dir).exists()
            and results_sha256(experiment_dir) == rebuilt_sha256
        ):
            return f"The leaderboard already matches {len(rebuilt)} recorded candidates."
        selection_path = experiment_dir / SELECTION_PATH
        if selection_path.is_file():
            selection = FrozenSelection.model_validate_json(read_text(selection_path))
            if selection.results_sha256 != rebuilt_sha256:
                raise ValueError(
                    "Selection is frozen, and the run records give a different leaderboard. "
                    "Nothing was rebuilt."
                )
        if not results_path(experiment_dir).exists():
            write_results(experiment_dir, rebuilt)
            return f"Rebuilt the missing leaderboard from {len(rebuilt)} recorded candidates."
        try:
            changes = _describe_changes(load_results(experiment_dir), rebuilt)
        except ValueError:
            write_results(experiment_dir, rebuilt)
            return (
                f"Rebuilt the leaderboard from {len(rebuilt)} recorded candidates. "
                "The old file was invalid, so what it held is unknown."
            )
        summary = " ".join(changes) or "Only the order changed."
        if changes and change_reason is None:
            raise ValueError(
                f"The run records disagree with the leaderboard. {summary} Nothing was rebuilt. "
                "If these are right, pass `--allow-changes --reason <text>`, which labels the "
                "result exploratory."
            )
        if changes:
            path = experiment_dir / REBUILDS_PATH
            earlier: list[str] = json.loads(read_text(path)) if path.is_file() else []
            write_json(path, [*earlier, change_reason])
        write_results(experiment_dir, rebuilt)
        return f"Rebuilt the leaderboard from {len(rebuilt)} recorded candidates. {summary}"


def reseal_holdout(experiment_dir: Path, rows: pd.DataFrame, provenance_note: str) -> HoldoutSeal:
    """Seal new rows, retire any current holdout, and reopen search.

    With no current holdout this adds the first one. Validation runs before the first change. The
    ledger entry is the first change. A retry after a crash resumes, and a retry after the new seal
    committed finds those rows live and does nothing.
    """
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    with results_lock(experiment_dir):
        check_replacement_rows(experiment_dir, rows)
        if not (experiment_dir / HOLDOUT_SEAL_FILE_NAME).is_file():
            seal, staged = stage_replacement_rows(experiment_dir, str(uuid4()), rows)
            place_staged_file(seal, staged)
            commit_seal(experiment_dir, seal)
            return seal
        live = read_holdout_seal(experiment_dir)
        record = ConsumedHoldout(
            holdout=live,
            provenance_note=provenance_note,
            exploratory_run_ids=[result.run_id for result in load_results(experiment_dir)],
            resealed_at=utc_now(),
        )
        seal, staged = stage_replacement_rows(experiment_dir, live.experiment_uid, rows)
        if seal == live:
            staged.unlink(missing_ok=True)
            if entries_for(live.holdout_sha256):
                raise ValueError("These rows are the current holdout, and it was already read.")
            remove_other_sealed_files(live)
            return live

        record_retirement(
            LedgerEntry(
                holdout_sha256=live.holdout_sha256,
                event="retired",
                experiment_uid=live.experiment_uid,
                experiment_name=experiment_dir.resolve().name,
                experiment_path=repository_relative_path(experiment_dir),
                provenance_note=provenance_note,
                recorded_at=utc_now(),
            )
        )
        retired_before = history_seals(experiment_dir)
        if live.holdout_sha256 not in retired_before:
            _write_history(experiment_dir, len(retired_before) + 1, record)
        for relative in (SELECTION_PATH, ATTEMPT_PATH, FINAL_PATH):
            (experiment_dir / relative).unlink(missing_ok=True)
        place_staged_file(seal, staged)
        commit_seal(experiment_dir, seal)
        remove_other_sealed_files(seal)
        return seal
