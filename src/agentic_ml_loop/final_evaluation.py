"""Freeze a selection, then evaluate it once on the sealed holdout."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

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
    read_holdout_seal,
    require_history_in_ledger,
    seal_replacement_rows,
    sealed_file_path,
    verify_sealed_holdout,
)
from agentic_ml_loop.io import file_sha256, json_text, read_text, utc_now, write_json
from agentic_ml_loop.ledger import (
    LedgerEntry,
    ProvenanceNote,
    claim_for_evaluation,
    entries_for,
    record_retirement,
)
from agentic_ml_loop.paths import ATTEMPT_PATH, FINAL_PATH, HOLDOUT_HISTORY_DIR, SELECTION_PATH
from agentic_ml_loop.results import (
    RecordedResult,
    comparison_errors,
    load_results,
    results_lock,
    results_sha256,
)
from agentic_ml_loop.run_evidence import (
    CodeFingerprint,
    RunEvidence,
    recorded_fingerprint_is_current,
    validate_run_evidence,
)
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
    experiment_dir: Path, candidate_id: str, target_column: str
) -> FrozenSelection:
    """Bind one verified candidate before any holdout metric, and report split drift."""
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
        audit = audit_search_splits(load_search_splits(experiment_dir), target_column)
        selection = FrozenSelection(
            selected_result=selected,
            results_sha256=results_sha256(experiment_dir),
            code_fingerprint=_recorded_fingerprint(experiment_dir, selected),
            data_audit=audit,
        )
        path = experiment_dir / SELECTION_PATH
        if path.exists():
            if FrozenSelection.model_validate_json(read_text(path)) != selection:
                raise ValueError(
                    "Selection is already frozen. Preserve it and start a new experiment."
                )
            return selection
        if (experiment_dir / ATTEMPT_PATH).exists() or (experiment_dir / FINAL_PATH).exists():
            raise ValueError("Final evaluation was already attempted. Restore its selection.")
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
    if not recorded_fingerprint_is_current(experiment_dir, selection.code_fingerprint):
        raise ValueError("Runtime code changed after selection was frozen")
    return selection


def _require_ledger_chain(final: FinalEvaluation, attempt_sha256: str) -> None:
    """A read leaves a ledger entry first. Final evidence without one was not made by the engine."""
    if final.attempt_sha256 != attempt_sha256:
        raise ValueError("Final evidence does not chain to the attempt file")
    if not any(
        entry.event == "evaluated"
        and entry.experiment_uid == final.holdout.experiment_uid
        and entry.attempt_sha256 == attempt_sha256
        for entry in entries_for(final.holdout.holdout_sha256)
    ):
        raise ValueError("The ledger has no evaluation of this holdout that matches the attempt")


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
    _require_ledger_chain(final, file_sha256(experiment_dir / ATTEMPT_PATH))
    return final


def _check_rerun_matches(selected: RecordedResult, rerun: CandidateResult) -> None:
    for name in _FROZEN_FIELDS:
        if getattr(rerun, name) != getattr(selected, name):
            raise ValueError(f"Final rerun changed frozen selection field: {name}")


def evaluate_final(
    experiment_dir: Path, candidate_runners: Mapping[str, CandidateRunner]
) -> FinalEvaluation:
    """Check the frozen evidence and the seal hash, then score the sealed rows once."""
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    with results_lock(experiment_dir):
        selection = _load_selection(experiment_dir)
        selected = selection.selected_result
        selection_sha256 = file_sha256(experiment_dir / SELECTION_PATH)
        seal = read_holdout_seal(experiment_dir)
        attempt_path = experiment_dir / ATTEMPT_PATH
        final_path = experiment_dir / FINAL_PATH

        if attempt_path.exists():
            attempt = FinalAttempt.model_validate_json(read_text(attempt_path))
            if attempt.selection_sha256 != selection_sha256 or attempt.holdout != seal:
                raise ValueError("Final attempt does not match the frozen selection or the seal")
            if not final_path.exists():
                raise ValueError(
                    "Final evaluation was attempted but did not finish. "
                    "Inspect its artifacts before any explicit retry."
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
        claim_for_evaluation(
            LedgerEntry(
                holdout_sha256=seal.holdout_sha256,
                event="evaluated",
                experiment_uid=seal.experiment_uid,
                experiment_name=experiment_dir.resolve().name,
                attempt_sha256=attempt_sha256,
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
            ],
        )
        write_json(final_path, final.model_dump(mode="json"))
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


def reseal_holdout(experiment_dir: Path, rows: pd.DataFrame, provenance_note: str) -> HoldoutSeal:
    """Retire the current holdout, seal new rows, and reopen search.

    A holdout that was read cannot be made unbiased again. This records that fact with a required
    note. The ledger entry comes first: from then on evaluation refuses the old file, so a crash
    before the new seal commits leaves no holdout that can be read. A retry resumes.
    """
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    with results_lock(experiment_dir):
        retired = read_holdout_seal(experiment_dir)
        record = ConsumedHoldout(
            holdout=retired,
            provenance_note=provenance_note,
            exploratory_run_ids=[result.run_id for result in load_results(experiment_dir)],
            resealed_at=utc_now(),
        )
        require_history_in_ledger(experiment_dir)
        check_replacement_rows(experiment_dir, rows)
        record_retirement(
            LedgerEntry(
                holdout_sha256=retired.holdout_sha256,
                event="retired",
                experiment_uid=retired.experiment_uid,
                experiment_name=experiment_dir.resolve().name,
                provenance_note=provenance_note,
                recorded_at=utc_now(),
            )
        )
        retired_before = history_seals(experiment_dir)
        if not retired_before or retired_before[-1] != retired.holdout_sha256:
            _write_history(experiment_dir, len(retired_before) + 1, record)
        for relative in (SELECTION_PATH, ATTEMPT_PATH, FINAL_PATH):
            (experiment_dir / relative).unlink(missing_ok=True)
        seal = seal_replacement_rows(experiment_dir, retired, rows)
        commit_seal(experiment_dir, seal)
        sealed_file_path(retired).unlink(missing_ok=True)
        return seal
