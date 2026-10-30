"""Seal holdout rows outside the workspace and record only their identity, hash, and row count."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from agentic_ml_loop.io import file_sha256, read_text, write_json
from agentic_ml_loop.ledger import entries_for, read_ledger, store_lock, store_root
from agentic_ml_loop.paths import (
    DATA_DIRNAME,
    HOLDOUT_HISTORY_DIR,
    HOLDOUT_SEAL_FILE_NAME,
    SNAPSHOT_FILE_NAME,
    data_dir,
)
from agentic_ml_loop.splits import HoldoutSplits, SearchSplits, Sha256

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


def _write_sealed_file(
    rows: pd.DataFrame, experiment_uid: str, experiment_dir: Path
) -> HoldoutSeal:
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
        known = entries_for(seal.holdout_sha256)
        if known:
            raise ValueError(
                f"This exact file was already {known[0].event} by experiment "
                f"{known[0].experiment_name}. It cannot give a fresh estimate. Use unseen rows."
            )
        candidate.chmod(0o600)
        candidate.replace(sealed_file_path(seal))
    return seal


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


def consumed_holdout_count(experiment_dir: Path) -> int:
    return len(history_seals(experiment_dir))


def require_history_in_ledger(experiment_dir: Path) -> None:
    known = {entry.holdout_sha256 for entry in read_ledger()}
    missing = [sha for sha in history_seals(experiment_dir) if sha not in known]
    if missing:
        raise ValueError(
            f"The holdout ledger lacks {len(missing)} holdouts this experiment retired. "
            "It was deleted or replaced, so no new seal can be checked. Restore it."
        )


def seal_holdout(experiment_dir: Path, search: SearchSplits, holdout: HoldoutSplits) -> HoldoutSeal:
    """Write the holdout outside the workspace and leave search rows in ``data/``."""
    record_path = experiment_dir / HOLDOUT_SEAL_FILE_NAME
    if holdout.holdout.empty or search.train.empty or search.validation.empty:
        raise ValueError("Train, validation, and holdout rows must all be non-empty")
    with store_lock():
        if record_path.exists():
            raise ValueError(f"Holdout is already sealed: {record_path}")
        require_history_in_ledger(experiment_dir)
        seal = _write_sealed_file(holdout.holdout, str(uuid4()), experiment_dir)
        rows_dir = data_dir(experiment_dir)
        _write_rows(search.train, rows_dir / TRAIN_FILE_NAME)
        _write_rows(search.validation, rows_dir / VALIDATION_FILE_NAME)
        write_json(record_path, seal.model_dump())
    return seal


def load_search_splits(experiment_dir: Path) -> SearchSplits:
    rows_dir = experiment_dir / DATA_DIRNAME
    if (rows_dir / SNAPSHOT_FILE_NAME).exists() and (
        experiment_dir / HOLDOUT_SEAL_FILE_NAME
    ).exists():
        raise ValueError("A full snapshot sits beside a seal. Run `experiment seal` to finish it.")
    train_path = rows_dir / TRAIN_FILE_NAME
    validation_path = rows_dir / VALIDATION_FILE_NAME
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(f"No search rows in {rows_dir}. Seal the holdout first.")
    return SearchSplits(
        train=pd.read_parquet(train_path), validation=pd.read_parquet(validation_path)
    )


def read_holdout_seal(experiment_dir: Path) -> HoldoutSeal:
    path = experiment_dir / HOLDOUT_SEAL_FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No holdout seal at {path}. Seal the holdout first.")
    return HoldoutSeal.model_validate_json(read_text(path))


def verify_sealed_holdout(seal: HoldoutSeal) -> None:
    """Compare file bytes to the seal without parsing a row."""
    path = sealed_file_path(seal)
    if not path.is_file():
        raise FileNotFoundError(f"Sealed holdout is missing from the store: {seal.experiment_uid}")
    if file_sha256(path) != seal.holdout_sha256:
        raise ValueError(f"Sealed holdout changed after sealing: {seal.experiment_uid}")


def load_sealed_holdout(seal: HoldoutSeal) -> HoldoutSplits:
    verify_sealed_holdout(seal)
    holdout = pd.read_parquet(sealed_file_path(seal))
    if len(holdout) != seal.holdout_rows:
        raise ValueError(f"Sealed holdout row count changed: {seal.experiment_uid}")
    return HoldoutSplits(holdout=holdout)


def check_replacement_rows(experiment_dir: Path, rows: pd.DataFrame) -> None:
    if rows.empty:
        raise ValueError("New holdout rows must be non-empty")
    search = load_search_splits(experiment_dir)
    if {str(column) for column in rows.columns} != {str(column) for column in search.train.columns}:
        raise ValueError("New holdout rows must have the same columns as the search rows")


def seal_replacement_rows(
    experiment_dir: Path, retiring: HoldoutSeal, rows: pd.DataFrame
) -> HoldoutSeal:
    """Store new rows without committing them. The ledger refuses an exact file it knows."""
    return _write_sealed_file(rows, retiring.experiment_uid, experiment_dir)


def commit_seal(experiment_dir: Path, seal: HoldoutSeal) -> None:
    write_json(experiment_dir / HOLDOUT_SEAL_FILE_NAME, seal.model_dump())
