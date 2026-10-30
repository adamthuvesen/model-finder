"""One rendering of expected failures: a concise stderr line and exit code 1."""

from __future__ import annotations

import sys
from collections.abc import Callable

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


def run_reporting_errors(command: Callable[[], int]) -> int:
    try:
        return command()
    except EXPECTED_ERRORS as exc:
        print(f"error: {describe(exc)}", file=sys.stderr)
        return 1
