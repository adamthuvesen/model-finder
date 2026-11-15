"""What the post-edit hook needs in Python. It imports only the standard library, so it starts fast."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import cast

REPORT_LIMIT = 2000
SEE_ALL = ".venv/bin/ruff check . && .venv/bin/basedpyright --level error"
_PATCHED_FILE = re.compile(r"^\*\*\* (?:Update|Add) File: (.+)$", re.MULTILINE)


def edited_files(event: dict[str, object], repo_root: Path) -> list[str]:
    """Read the edited paths from any client's hook input. Other folders are not this repo's."""
    cwd = event.get("cwd")
    base = Path(cwd) if isinstance(cwd, str) else repo_root
    root = repo_root.resolve()
    named = _named_files(event)
    resolved = [(base / name).resolve() for name in named]
    return [str(path) for path in resolved if path.is_relative_to(root)]


def _named_files(event: dict[str, object]) -> list[str]:
    tool_input = event.get("tool_input")
    fields = cast("dict[str, object]", tool_input) if isinstance(tool_input, dict) else {}
    for container in (event, fields):
        for key in ("file_path", "path"):
            value = container.get(key)
            if isinstance(value, str):
                return [value]
    patch = fields.get("command")
    return _PATCHED_FILE.findall(patch) if isinstance(patch, str) else []


def experiment_of(file: Path) -> Path | None:
    """Return the experiment that holds a file. Its candidate code is checked as Python instead."""
    for parent in file.resolve().parents:
        if (parent / "experiment.md").is_file():
            return parent
    return None


def report(client: str, findings: str) -> str:
    """Hand findings to the agent in the client's documented shape."""
    if len(findings) > REPORT_LIMIT:
        findings = (
            f"{findings[:REPORT_LIMIT]}\n{len(findings) - REPORT_LIMIT} more characters. "
            f"See all of it: {SEE_ALL}"
        )
    if client == "cursor":
        return json.dumps({"additional_context": findings})
    return json.dumps(
        {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": findings}}
    )


def main(arguments: list[str]) -> int:
    action = arguments[0]
    if action == "files":
        try:
            event: object = json.loads(sys.stdin.read() or "{}")
        except json.JSONDecodeError:
            return 0
        if isinstance(event, dict):
            print("\n".join(edited_files(cast("dict[str, object]", event), Path.cwd())))
    elif action == "experiment":
        found = experiment_of(Path(arguments[1]))
        if found is not None:
            print(found)
    elif action == "report":
        findings = sys.stdin.read().strip()
        if findings:
            print(report(arguments[1], findings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
