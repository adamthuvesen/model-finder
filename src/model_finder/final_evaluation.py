"""Freeze a selection, then evaluate it once on the sealed holdout."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from model_finder.candidate_result import (
    CandidateResult,
    CandidateRunner,
    MetricValues,
    SearchMetrics,
    run_candidate,
)
from model_finder.data_audit import (
    DataAudit,
    OverlapReport,
    audit_search_splits,
    overlap_report,
    target_shift_concern,
)
from model_finder.holdout import (
    HoldoutSeal,
    consumed_holdout_count,
    holdout_read_labels,
    load_sealed_holdout,
    load_search_splits,
    read_holdout_seal,
    replace_read_evidence,
    verify_sealed_holdout,
    withdraw_selection,
)
from model_finder.io import file_sha256, json_text, read_text, utc_now, write_json
from model_finder.ledger import (
    LedgerEntry,
    claim_for_evaluation,
    entries_for,
    record_final,
)
from model_finder.paths import (
    ATTEMPT_PATH,
    FINAL_PATH,
    SELECTION_PATH,
    repository_relative_path,
    require_experiment_dir,
)
from model_finder.results import (
    RecordedResult,
    comparison_errors,
    load_results,
    results_lock,
    results_sha256,
)
from model_finder.run_evidence import (
    CodeFingerprint,
    RunEvidence,
    run_evidence_staleness,
    validate_run_evidence,
)
from model_finder.spec import get_key_columns
from model_finder.splits import Sha256, search_data_identity

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


_SEARCH_DATA_CHANGED = (
    "data/: Search data changed after selection was frozen. "
    "Fix: restore the rows, or run the candidates again and freeze again"
)
_RUN_THE_FINAL_AGAIN = (
    "Fix: run `evaluate-final --allow-reread --reason <text>`, "
    "which writes real evidence and labels it exploratory"
)


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
    require_experiment_dir(experiment_dir)
    with results_lock(experiment_dir):
        results = load_results(experiment_dir)
        _raise_errors(comparison_errors(results))
        matches = [result for result in results if result.candidate_id == candidate_id]
        if len(matches) != 1:
            raise ValueError(
                f"results.json: no single candidate named {candidate_id}. "
                "Fix: run `experiment leaderboard` to see the names, or run the candidate"
            )
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
        raise ValueError(
            f"{path}: No frozen selection. Fix: run `freeze-candidate --candidate <id>` first"
        )
    selection = FrozenSelection.model_validate_json(read_text(path))
    _raise_errors(validate_run_evidence(experiment_dir, selection.selected_result))
    if selection.results_sha256 != results_sha256(experiment_dir):
        raise ValueError(
            "results.json: Results changed after selection was frozen. "
            "Fix: run `freeze-candidate --candidate <id>` again"
        )
    if search_data_identity(load_search_splits(experiment_dir)) != (
        selection.selected_result.comparison_context.data
    ):
        raise ValueError(_SEARCH_DATA_CHANGED)
    return selection


def _require_ledger_chain(
    experiment_dir: Path, final: FinalEvaluation, attempt_sha256: str
) -> None:
    """The engine records a read and its final file's hash. Other evidence was not made by it."""
    if final.attempt_sha256 != attempt_sha256:
        raise ValueError(
            f"{FINAL_PATH}: does not chain to the attempt file. {_RUN_THE_FINAL_AGAIN}"
        )
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
        raise ValueError(
            f"{ATTEMPT_PATH}: the ledger has no evaluation of this holdout that matches it. "
            f"{_RUN_THE_FINAL_AGAIN}"
        )
    final_sha256 = file_sha256(experiment_dir / FINAL_PATH)
    finalized = [
        entry
        for entry in entries
        if entry.event == "finalized"
        and entry.attempt_sha256 == attempt_sha256
        and entry.final_sha256 == final_sha256
    ]
    if not finalized:
        raise ValueError(
            f"{FINAL_PATH}: the ledger has no record of this file, so it was edited or forged. "
            f"{_RUN_THE_FINAL_AGAIN}"
        )


def load_verified_final(experiment_dir: Path) -> tuple[FrozenSelection, FinalEvaluation]:
    """Return final evidence only when it matches its frozen selection, attempt, and seal."""
    selection = _load_selection(experiment_dir)
    attempt_path = experiment_dir / ATTEMPT_PATH
    if not attempt_path.is_file() or not (experiment_dir / FINAL_PATH).is_file():
        raise ValueError(f"{FINAL_PATH}: No completed final evaluation. Fix: run `evaluate-final`")
    attempt = FinalAttempt.model_validate_json(read_text(attempt_path))
    if attempt.selection_sha256 != file_sha256(experiment_dir / SELECTION_PATH):
        raise ValueError(
            f"{ATTEMPT_PATH}: does not match the frozen selection. {_RUN_THE_FINAL_AGAIN}"
        )
    if attempt.holdout != read_holdout_seal(experiment_dir):
        raise ValueError(f"{ATTEMPT_PATH}: does not match the current seal. {_RUN_THE_FINAL_AGAIN}")
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
        raise ValueError(
            f"{FINAL_PATH}: does not match its selection or attempt. {_RUN_THE_FINAL_AGAIN}"
        )
    _require_ledger_chain(experiment_dir, final, file_sha256(experiment_dir / ATTEMPT_PATH))
    return final


def selection_staleness(experiment_dir: Path, selection: FrozenSelection) -> list[str]:
    return run_evidence_staleness(experiment_dir, selection.selected_result)


def _check_rerun_matches(selected: RecordedResult, rerun: CandidateResult) -> None:
    for name in _FROZEN_FIELDS:
        if getattr(rerun, name) != getattr(selected, name):
            raise ValueError(
                f"the final rerun changed frozen selection field {name}. "
                "Fix: run the candidate again, then freeze it again. No holdout row was read"
            )


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
    require_experiment_dir(experiment_dir)
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
                raise ValueError(
                    f"{ATTEMPT_PATH}: does not match the selection or the seal. {_RUN_THE_FINAL_AGAIN}"
                )
            if not final_path.exists():
                raise ValueError(
                    f"{ATTEMPT_PATH}: Final evaluation was attempted but did not finish. "
                    "Fix: run `evaluate-final --allow-reread --reason <text>`, which labels "
                    "the estimate exploratory"
                )
            return _completed_evaluation(experiment_dir, selection, attempt)
        if final_path.exists():
            raise ValueError(f"{FINAL_PATH}: has no recorded attempt. {_RUN_THE_FINAL_AGAIN}")
        if selected.candidate_id not in candidate_runners:
            raise ValueError(
                f"frozen candidate {selected.candidate_id} is not in this runner. "
                "Fix: add it back, or freeze another candidate"
            )

        verify_sealed_holdout(seal)
        search = load_search_splits(experiment_dir)
        if search_data_identity(search) != selected.comparison_context.data:
            raise ValueError(_SEARCH_DATA_CHANGED)

        fitted = run_candidate(candidate_runners[selected.candidate_id], search)
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
            raise ValueError(
                f"{SELECTION_PATH}: changed during final evaluation. {_RUN_THE_FINAL_AGAIN}"
            )
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
