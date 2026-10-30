"""Seal new rows, and retire any current holdout."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict

from model_finder.holdout import (
    HoldoutSeal,
    check_replacement_rows,
    commit_seal,
    history_seals,
    place_staged_file,
    read_holdout_seal,
    remove_other_sealed_files,
    stage_replacement_rows,
)
from model_finder.io import utc_now, write_json
from model_finder.ledger import LedgerEntry, ProvenanceNote, entries_for, record_retirement
from model_finder.paths import (
    ATTEMPT_PATH,
    FINAL_PATH,
    HOLDOUT_HISTORY_DIR,
    HOLDOUT_SEAL_FILE_NAME,
    SELECTION_PATH,
    repository_relative_path,
    require_experiment_dir,
)
from model_finder.results import load_results, results_lock


class ConsumedHoldout(BaseModel):
    """The permanent record that an earlier holdout no longer gives an unbiased estimate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    holdout: HoldoutSeal
    provenance_note: ProvenanceNote
    exploratory_run_ids: list[str]
    resealed_at: str


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
    """Seal new rows, retire any current holdout, and reopen search.

    With no current holdout this adds the first one. Validation runs before the first change. The
    ledger entry is the first change. A retry after a crash resumes, and a retry after the new seal
    committed finds those rows live and does nothing.
    """
    require_experiment_dir(experiment_dir)
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
                raise ValueError(
                    "These rows are the current holdout, and it was already read. "
                    "Fix: pass new rows, or use `evaluate-final --allow-reread --reason <text>`"
                )
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
