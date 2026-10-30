"""One rendering of expected failures: a concise stderr line and exit code 1."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from agentic_ml_loop.ledger import entries_for
from agentic_ml_loop.sources.errors import SourceError

EXPECTED_ERRORS = (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, SourceError)


LEADERBOARD_REPAIR = " Do not edit the leaderboard. Run `experiment rebuild-results`."


def describe(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'value'}: {error['msg']}"
            for error in exc.errors()
        )
        hint = LEADERBOARD_REPAIR if "RecordedResult" in exc.title else ""
        return f"invalid {exc.title}: {details}{hint}"
    return str(exc)


def warn_rows_stay_readable(source: Path) -> None:
    print(
        f"warning: Sealed rows stay readable in {source}. The engine cannot delete that file. "
        "Move it out of the agent's reach, or deny it in the agent's settings.",
        file=sys.stderr,
    )


def override_reason(flag: str, *, allowed: bool, reason: str | None) -> str | None:
    """An override needs its flag and a reason together, so neither is dropped silently."""
    if allowed and not (reason or "").strip():
        raise ValueError(f"{flag} needs --reason <text>")
    if reason is not None and not allowed:
        raise ValueError(f"--reason does nothing without {flag}")
    return reason if allowed else None


def warn_if_already_read(holdout_sha256: str) -> None:
    known = entries_for(holdout_sha256)
    if known:
        print(
            f"warning: This exact file was already {known[0].event} by experiment "
            f"{known[0].experiment_name}. Its estimate will be labelled exploratory.",
            file=sys.stderr,
        )


def run_reporting_errors(command: Callable[[], int]) -> int:
    try:
        return command()
    except EXPECTED_ERRORS as exc:
        print(f"error: {describe(exc)}", file=sys.stderr)
        return 1
