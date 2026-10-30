"""The pre-tool hook for Codex and Cursor: refuse the holdout store and declared connector writes.

Claude Code gets the same limits as deny rules in `.claude/settings.json`. Both come from
`agent_rules`, so there is one list. Like the Claude rules, this matches names in the call, so a
path that a script builds, or a `cd` into the store, is not caught.

Usage: `python -m model_finder.cli.guard <codex|cursor> <repo root>`, with the hook input on stdin.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import cast

from model_finder.cli.agent_rules import HOLDOUT_STORE_MARKER, declared_connectors

_PATCHED_PATH = re.compile(r"^\*\*\* (?:(?:Add|Update|Delete) File|Move to): (.+)$", re.MULTILINE)
HOLDOUT_REASON = (
    "the sealed holdout store is off limits to agents. "
    "Fix: score the frozen candidate with `experiment evaluate-final`"
)
CONNECTOR_REASON = (
    "{tool} is a declared connector write tool, and agents do not write to external systems. "
    "Fix: ask the person to make this change"
)


def _named_text(payload: dict[str, object]) -> list[str]:
    """The parts of a call that name what it touches. File contents and patch bodies do not count."""
    tool_input = payload.get("tool_input")
    fields = cast("dict[str, object]", tool_input) if isinstance(tool_input, dict) else {}
    if payload.get("tool_name") == "apply_patch":
        patch = fields.get("command") or fields.get("input") or ""
        return _PATCHED_PATH.findall(patch) if isinstance(patch, str) else []
    named = [payload.get("command"), payload.get("file_path"), *fields.values()]
    return [value if isinstance(value, str) else json.dumps(value) for value in named if value]


def refusal(payload: dict[str, object], repo_root: Path) -> str | None:
    """Return why the call is refused, or None to allow it."""
    if any(HOLDOUT_STORE_MARKER in text for text in _named_text(payload)):
        return HOLDOUT_REASON
    tool = payload.get("tool_name")
    is_mcp = payload.get("hook_event_name") == "beforeMCPExecution" or (
        isinstance(tool, str) and tool.startswith("mcp__")
    )
    if is_mcp and isinstance(tool, str):
        write_tools = {
            write_tool
            for connector in declared_connectors(repo_root) or []
            for write_tool in connector.write_tools
        }
        if tool.rsplit("__", 1)[-1] in write_tools:
            return CONNECTOR_REASON.format(tool=tool)
    return None


def answer(client: str, reason: str | None) -> str:
    """The reply in the client's hook format. Codex allows on no output; Cursor needs a reply."""
    if client == "cursor":
        if reason is None:
            return json.dumps({"permission": "allow"})
        return json.dumps(
            {"permission": "deny", "user_message": f"Blocked: {reason}", "agent_message": reason}
        )
    if reason is None:
        return ""
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
    )


def main(arguments: list[str]) -> int:
    try:
        client, root = arguments
        raw = sys.stdin.read().strip()
        payload = cast("dict[str, object]", json.loads(raw) if raw else {})
        reply = answer(client, refusal(payload, Path(root)))
    except Exception as exc:  # noqa: BLE001 - a broken guard must block the call, not allow it
        print(f"model-finder guard failed, so the call is blocked: {exc}", file=sys.stderr)
        return 2
    if reply:
        print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
