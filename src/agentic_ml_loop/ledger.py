"""An append-only record, across all experiments, of holdouts that were read or retired.

A holdout's identity is the sha256 of its sealed file. The engine enforces the workflow around that
file. It does not try to recognize similar data: a resplit or a schema change makes a new file, and
so a new holdout. The ledger is a plain file, so an agent with a shell can edit it.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_ml_loop.splits import Sha256

_LEDGER_FILE_NAME = "ledger.jsonl"
ProvenanceNote = Annotated[str, Field(min_length=1, pattern=r"\S")]


class LedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    holdout_sha256: Sha256
    event: Literal["evaluated", "finalized", "retired"]
    experiment_uid: str
    experiment_name: str
    experiment_path: str
    attempt_sha256: Sha256 | None = None
    final_sha256: Sha256 | None = None
    provenance_note: ProvenanceNote | None = None
    recorded_at: str


def store_root() -> Path:
    configured = os.environ.get("XDG_DATA_HOME", "")
    root = Path(configured) if Path(configured).is_absolute() else Path.home() / ".local" / "share"
    return root / "agentic-ml-loop" / "holdouts"


def _ledger_path() -> Path:
    root = store_root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root / _LEDGER_FILE_NAME


@contextmanager
def _exclusive(name: str) -> Generator[None]:
    root = store_root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def store_lock() -> AbstractContextManager[None]:
    """Serialize seals, so two concurrent seals cannot mix their files in one workspace."""
    return _exclusive("seal.lock")


def read_ledger() -> list[LedgerEntry]:
    """Return every entry. An unreadable store or a corrupt line fails closed."""
    path = _ledger_path()
    if not path.exists():
        return []
    entries: list[LedgerEntry] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entries.append(LedgerEntry.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(
                    f"Holdout ledger {path} is corrupt at line {number}. Every experiment is "
                    "blocked until that line is repaired, because a used holdout cannot be ruled out."
                ) from exc
    return entries


def entries_for(holdout_sha256: str) -> list[LedgerEntry]:
    return [entry for entry in read_ledger() if entry.holdout_sha256 == holdout_sha256]


def _append(entry: LedgerEntry) -> None:
    path = _ledger_path()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


def claim_for_evaluation(entry: LedgerEntry) -> None:
    """Record the read before it happens. Any entry for this file, anywhere, refuses it."""
    with _exclusive("ledger.lock"):
        known = entries_for(entry.holdout_sha256)
        if known:
            first = known[0]
            raise ValueError(
                f"This holdout file was already {first.event} by experiment "
                f"{first.experiment_name} ({first.experiment_uid}). It cannot give a fresh "
                "estimate. Seal unseen rows with `experiment reseal`."
            )
        _append(entry)


def record_retirement(entry: LedgerEntry) -> None:
    """Append once per experiment, so a retried reseal keeps its first note."""
    with _exclusive("ledger.lock"):
        known = entries_for(entry.holdout_sha256)
        if not any(
            known_entry.event == "retired" and known_entry.experiment_uid == entry.experiment_uid
            for known_entry in known
        ):
            _append(entry)


def record_final(entry: LedgerEntry) -> None:
    """Record the final file's hash before the file exists. It binds the only valid final."""
    with _exclusive("ledger.lock"):
        _append(entry)


def retired_count(experiment_uid: str) -> int:
    """Count retired holdouts of one experiment. Every kill point in a reseal gives one answer."""
    return len(
        {
            entry.holdout_sha256
            for entry in read_ledger()
            if entry.event == "retired" and entry.experiment_uid == experiment_uid
        }
    )
