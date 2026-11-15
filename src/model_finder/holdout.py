"""Seal holdout rows outside the workspace and record only their identity, hash, and row count."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from model_finder.io import file_sha256, read_text, utc_now, write_json
from model_finder.ledger import (
    LedgerEntry,
    entries_for,
    record_reopened,
    retired_count,
    store_lock,
    store_root,
)
from model_finder.paths import (
    ATTEMPT_PATH,
    DATA_DIRNAME,
    FINAL_PATH,
    HOLDOUT_HISTORY_DIR,
    HOLDOUT_SEAL_FILE_NAME,
    REREADS_DIR,
    SELECTION_PATH,
    SNAPSHOT_FILE_NAME,
    data_dir,
    repository_relative_path,
)
from model_finder.splits import HoldoutSplits, SearchSplits, Sha256

TRAIN_FILE_NAME = "train.parquet"
VALIDATION_FILE_NAME = "validation.parquet"


class HoldoutSeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_uid: Annotated[str, Field(pattern=r"^[0-9a-f-]{36}$")]
    holdout_sha256: Sha256
    holdout_rows: Annotated[int, Field(gt=0)]


def sealed_file_path(seal: HoldoutSeal) -> Path:
    return store_root() / seal.experiment_uid / f"holdout-{seal.holdout_sha256}.parquet"


def _stage_sealed_file(
    rows: pd.DataFrame, experiment_uid: str, experiment_dir: Path
) -> tuple[HoldoutSeal, Path]:
    """Write rows to the store under a staging name and return their identity."""
    directory = store_root() / experiment_uid
    if directory.resolve().is_relative_to(experiment_dir.resolve()):
        raise ValueError("The holdout store must stay outside the experiment directory")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        candidate = Path(scratch) / "holdout.parquet"
        rows.to_parquet(candidate, index=False)
        seal = HoldoutSeal(
            experiment_uid=experiment_uid,
            holdout_sha256=file_sha256(candidate),
            holdout_rows=len(rows),
        )
        staged = directory / f"staged-{seal.holdout_sha256}.parquet"
        candidate.chmod(0o600)
        candidate.replace(staged)
    return seal, staged


def place_staged_file(seal: HoldoutSeal, staged: Path) -> None:
    staged.replace(sealed_file_path(seal))


def remove_other_sealed_files(seal: HoldoutSeal) -> None:
    """Delete files left by retired seals. A retry after a crash finishes this too."""
    for path in (store_root() / seal.experiment_uid).glob("*.parquet"):
        if path != sealed_file_path(seal):
            path.unlink(missing_ok=True)


def _write_rows(rows: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    rows.to_parquet(temporary, index=False)
    temporary.replace(path)


def history_seals(experiment_dir: Path) -> list[str]:
    """Return the hash of each retired holdout, oldest first."""
    history = experiment_dir / HOLDOUT_HISTORY_DIR
    if not history.is_dir():
        return []
    return [
        json.loads(read_text(entry / "consumed.json"))["holdout"]["holdout_sha256"]
        for entry in sorted(history.iterdir())
        if entry.is_dir() and (entry / "consumed.json").is_file()
    ]


def is_sealed(experiment_dir: Path) -> bool:
    return (experiment_dir / HOLDOUT_SEAL_FILE_NAME).is_file()


def consumed_holdout_count(experiment_dir: Path) -> int:
    """Count retired holdouts from the ledger, so workspace edits cannot change the answer."""
    if not is_sealed(experiment_dir):
        return 0
    return retired_count(read_holdout_seal(experiment_dir).experiment_uid)


def seal_holdout(experiment_dir: Path, search: SearchSplits, holdout: HoldoutSplits) -> HoldoutSeal:
    """Write the holdout outside the workspace and leave search rows in ``data/``."""
    record_path = experiment_dir / HOLDOUT_SEAL_FILE_NAME
    if holdout.holdout.empty or search.train.empty or search.validation.empty:
        raise ValueError("a split is empty. Fix: use more rows, or change the split fractions")
    with store_lock():
        if is_sealed(experiment_dir):
            raise ValueError(
                f"{record_path}: already sealed. Fix: add rows with `experiment reseal`"
            )
        seal, staged = _stage_sealed_file(holdout.holdout, str(uuid4()), experiment_dir)
        place_staged_file(seal, staged)
        rows_dir = data_dir(experiment_dir)
        _write_rows(search.train, rows_dir / TRAIN_FILE_NAME)
        _write_rows(search.validation, rows_dir / VALIDATION_FILE_NAME)
        write_json(record_path, seal.model_dump())
    return seal


def write_search_rows(experiment_dir: Path, search: SearchSplits) -> None:
    """Write search rows with no seal. Results stay labelled exploratory."""
    if search.train.empty or search.validation.empty:
        raise ValueError(
            "train or validation rows are empty. Fix: use more rows, or change the split fractions"
        )
    rows_dir = data_dir(experiment_dir)
    _write_rows(search.train, rows_dir / TRAIN_FILE_NAME)
    _write_rows(search.validation, rows_dir / VALIDATION_FILE_NAME)


def _archive_read(experiment_dir: Path) -> None:
    """Keep the evidence of a read that later work replaces."""
    root = experiment_dir / REREADS_DIR
    target = root / f"{sum(1 for _ in root.glob('[0-9]*')) + 1:04d}"
    target.mkdir(parents=True)
    for relative in (SELECTION_PATH, ATTEMPT_PATH, FINAL_PATH):
        evidence = experiment_dir / relative
        if evidence.exists():
            shutil.copy2(evidence, target / evidence.name)


def replace_read_evidence(experiment_dir: Path) -> None:
    """Archive the attempt and the final, and keep the selection, for a second read of it."""
    _archive_read(experiment_dir)
    (experiment_dir / ATTEMPT_PATH).unlink(missing_ok=True)
    (experiment_dir / FINAL_PATH).unlink(missing_ok=True)


def withdraw_selection(experiment_dir: Path, reread_reason: str | None) -> None:
    """Reopen search. Before a read this costs nothing. After a read it needs a recorded reason."""
    has_read = (experiment_dir / ATTEMPT_PATH).exists() or (experiment_dir / FINAL_PATH).exists()
    if has_read:
        if reread_reason is None:
            raise ValueError(
                f"{FINAL_PATH}: this selection's holdout was read, so further search is "
                "holdout-informed. Fix: seal unseen rows with `experiment reseal`, or pass "
                "`--allow-reread --reason <text>`, which labels the result exploratory"
            )
        seal = read_holdout_seal(experiment_dir)
        record_reopened(
            LedgerEntry(
                holdout_sha256=seal.holdout_sha256,
                event="reopened",
                experiment_uid=seal.experiment_uid,
                experiment_name=experiment_dir.resolve().name,
                experiment_path=repository_relative_path(experiment_dir),
                provenance_note=reread_reason,
                recorded_at=utc_now(),
            )
        )
        _archive_read(experiment_dir)
    for relative in (SELECTION_PATH, ATTEMPT_PATH, FINAL_PATH):
        (experiment_dir / relative).unlink(missing_ok=True)


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


def load_search_splits(experiment_dir: Path) -> SearchSplits:
    rows_dir = experiment_dir / DATA_DIRNAME
    if (rows_dir / SNAPSHOT_FILE_NAME).exists() and is_sealed(experiment_dir):
        raise ValueError(
            f"{rows_dir / SNAPSHOT_FILE_NAME}: A full snapshot sits beside a seal. "
            "Fix: run `experiment seal` to finish it"
        )
    train_path = rows_dir / TRAIN_FILE_NAME
    validation_path = rows_dir / VALIDATION_FILE_NAME
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(
            f"{rows_dir}: No search rows. Fix: run your runner's `init-demo`, or `experiment seal`"
        )
    return SearchSplits(
        train=pd.read_parquet(train_path), validation=pd.read_parquet(validation_path)
    )


def read_holdout_seal(experiment_dir: Path) -> HoldoutSeal:
    path = experiment_dir / HOLDOUT_SEAL_FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: No holdout seal. Fix: add one with "
            "`experiment reseal --experiment <dir> --rows <parquet> --note <text>`"
        )
    return HoldoutSeal.model_validate_json(read_text(path))


def verify_sealed_holdout(seal: HoldoutSeal) -> None:
    """Compare file bytes to the seal without parsing a row."""
    path = sealed_file_path(seal)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: Sealed holdout is missing from the store. "
            "Fix: restore the file, or seal new rows with `experiment reseal`"
        )
    if file_sha256(path) != seal.holdout_sha256:
        raise ValueError(
            f"{path}: Sealed holdout changed after sealing. "
            "Fix: restore the file, or seal new rows with `experiment reseal`"
        )


def load_sealed_holdout(seal: HoldoutSeal) -> HoldoutSplits:
    verify_sealed_holdout(seal)
    holdout = pd.read_parquet(sealed_file_path(seal))
    if len(holdout) != seal.holdout_rows:
        raise ValueError(
            f"{sealed_file_path(seal)}: Sealed holdout row count changed. "
            "Fix: seal new rows with `experiment reseal`"
        )
    return HoldoutSplits(holdout=holdout)


def check_replacement_rows(experiment_dir: Path, rows: pd.DataFrame) -> None:
    if rows.empty:
        raise ValueError("New holdout rows must be non-empty. Fix: pass a file that has rows")
    search = load_search_splits(experiment_dir)
    if {str(column) for column in rows.columns} != {str(column) for column in search.train.columns}:
        raise ValueError(
            "New holdout rows must have the same columns as the search rows. "
            f"Fix: use these columns: {sorted(str(column) for column in search.train.columns)}"
        )


def stage_replacement_rows(
    experiment_dir: Path, experiment_uid: str, rows: pd.DataFrame
) -> tuple[HoldoutSeal, Path]:
    """Store new rows under a staging name. Nothing the experiment reads has changed yet."""
    return _stage_sealed_file(rows, experiment_uid, experiment_dir)


def commit_seal(experiment_dir: Path, seal: HoldoutSeal) -> None:
    write_json(experiment_dir / HOLDOUT_SEAL_FILE_NAME, seal.model_dump())
