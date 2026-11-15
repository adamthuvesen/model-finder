"""One rendering of expected failures: a concise stderr line and exit code 1."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from model_finder.ledger import entries_for
from model_finder.sources.errors import SourceError

EXPECTED_ERRORS = (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, SourceError)


_ENGINE_ROOT = Path(__file__).resolve().parents[1]
_NAMES_A_FILE = re.compile(r"^[^\s:]+(:\d+)?: ")
_EXPLANATIONS = {
    "missing": "{field} is required. Fix: set {field}",
    "extra_forbidden": "{field} is not a known field. Fix: remove it, or put it in hyperparameters",
    "finite_number": (
        "{field} is not a finite number. "
        "Fix: check the predictions and the target column for missing values"
    ),
    "json_invalid": "the file is not valid JSON. Fix: restore it",
    "string_pattern_mismatch": "{field} must not be blank. Fix: pass text",
    "string_too_short": "{field} must not be blank. Fix: pass text",
    "dict_type": "{field} must be a dict of names to numbers. Fix: pass a dict",
    "too_short": "{field} must not be empty. Fix: add at least one value",
}
_LEADERBOARD_FIX = "Fix: do not edit the leaderboard. Run `experiment rebuild-results <dir>`"


_KNOWN_FIXES: dict[type[BaseException], str] = {
    SyntaxError: "correct the syntax on that line",
    ZeroDivisionError: "guard the divisor on that line",
    AttributeError: "check the name on that line",
    IndexError: "check the length of what that line reads",
    NameError: "define or import the name on that line",
}


def _shown(path: Path) -> Path:
    return path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path


def _is_user_code(path: Path) -> bool:
    if not path.is_file() or "site-packages" in path.parts:
        return False
    return not path.resolve().is_relative_to(_ENGINE_ROOT)


def _user_location(exc: BaseException) -> str | None:
    """Name the last frame in the person's code, because that is the line to change."""
    if isinstance(exc, SyntaxError) and exc.filename and _is_user_code(Path(exc.filename)):
        return f"{_shown(Path(exc.filename).resolve())}:{exc.lineno}"
    if exc.__cause__ is not None:
        cause = _user_location(exc.__cause__)
        if cause is not None:
            return cause
    location: str | None = None
    traceback = exc.__traceback__
    while traceback is not None:
        path = Path(traceback.tb_frame.f_code.co_filename)
        if _is_user_code(path):
            location = f"{_shown(path.resolve())}:{traceback.tb_lineno}"
        traceback = traceback.tb_next
    return location


def _explain(error: Mapping[str, Any]) -> str:
    field = ".".join(str(part) for part in error["loc"]) or "value"
    if error["type"] == "value_error":
        message = error["msg"].removeprefix("Value error, ")
        return message if "Fix:" in message else f"{field}: {message}. Fix: correct {field}"
    template = _EXPLANATIONS.get(error["type"])
    if template:
        return template.format(field=field)
    return f"{field}: {error['msg']}. Fix: pass the expected type"


def describe(exc: BaseException) -> str:
    """Render one failure as ``<file>:<line>: <what is wrong>. Fix: <change or command>``."""
    location = _user_location(exc)
    prefix = f"{location}: " if location else ""
    if isinstance(exc, ValidationError):
        if "RecordedResult" in exc.title:
            problems = "; ".join(_explain(error).split(". Fix:")[0] for error in exc.errors())
            return f"results.json: {problems}. {_LEADERBOARD_FIX}"
        problems = dict.fromkeys(_explain(error) for error in exc.errors())
        return "\n".join(f"{prefix}{problem}" for problem in problems)
    message = exc.msg if isinstance(exc, SyntaxError) else str(exc)
    if "Fix:" in message:
        return message if _NAMES_A_FILE.match(message) else f"{prefix}{message}"
    if isinstance(exc, FileNotFoundError) and exc.filename:
        return f"{exc.filename}: no such file. Fix: correct the path"
    if location is None:
        return message
    known = next((fix for kind, fix in _KNOWN_FIXES.items() if isinstance(exc, kind)), None)
    ending = f". Fix: {known}" if known else ""
    return f"{location}: {type(exc).__name__}: {message}{ending}"


def warn_rows_stay_readable(source: Path) -> None:
    print(
        f"warning: Sealed rows stay readable in {source}. The engine cannot delete that file. "
        "Move it out of the agent's reach, or deny it in the agent's settings.",
        file=sys.stderr,
    )


def override_reason(flag: str, *, allowed: bool, reason: str | None) -> str | None:
    """An override needs its flag and a reason together, so neither is dropped silently."""
    if allowed and not (reason or "").strip():
        raise ValueError(f"{flag} needs --reason <text>. Fix: add --reason with one line")
    if reason is not None and not allowed:
        raise ValueError(
            f"--reason does nothing without {flag}. Fix: add {flag}, or remove --reason"
        )
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
    """Report what the person can fix. A bug in the engine itself keeps its traceback."""
    try:
        return command()
    except Exception as exc:
        if not isinstance(exc, EXPECTED_ERRORS) and _user_location(exc) is None:
            raise
        print(describe(exc), file=sys.stderr)
        return 1
