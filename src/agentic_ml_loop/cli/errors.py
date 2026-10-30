"""One rendering of expected failures: a concise stderr line and exit code 1."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from agentic_ml_loop.sources.errors import SourceError

EXPECTED_ERRORS = (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, SourceError)


def describe(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'value'}: {error['msg']}"
            for error in exc.errors()
        )
        return f"invalid {exc.title}: {details}"
    return str(exc)


def warn_rows_stay_readable(source: Path) -> None:
    print(
        f"warning: Sealed rows stay readable in {source}. The engine cannot delete that file. "
        "Move it out of the agent's reach, or deny it in the agent's settings.",
        file=sys.stderr,
    )


def run_reporting_errors(command: Callable[[], int]) -> int:
    try:
        return command()
    except EXPECTED_ERRORS as exc:
        print(f"error: {describe(exc)}", file=sys.stderr)
        return 1
