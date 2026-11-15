from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from model_finder.cli import guard
from model_finder.cli.main import main
from model_finder.cli.setup import GUARD_HOOKS, HOOK_SCRIPT

STORE_FILE = "~/.local/share/model-finder/holdouts/abc/holdout.parquet"
CONNECTORS = (
    '[[tool.model-finder.connectors]]\nname = "Slack"\nwrite_tools = ["slack_send_message"]\n'
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(CONNECTORS)
    return tmp_path


def _reply(client: str, payload: dict[str, object], repo: Path, monkeypatch, capsys) -> str:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert guard.main([client, str(repo)]) == 0
    return capsys.readouterr().out.strip()


def _codex(tool: str, tool_input: dict[str, object]) -> dict[str, object]:
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input}


@pytest.mark.parametrize(
    "payload",
    [
        _codex("Bash", {"command": f"cat {STORE_FILE}"}),
        _codex("apply_patch", {"command": f"*** Begin Patch\n*** Add File: {STORE_FILE}\n+x\n"}),
        _codex("mcp__claude_ai_Slack__slack_send_message", {"text": "hi"}),
    ],
    ids=["shell-read", "patch-into-store", "connector-write"],
)
def test_codex_denies(payload, repo: Path, monkeypatch, capsys) -> None:
    out = json.loads(_reply("codex", payload, repo, monkeypatch, capsys))["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert "Fix:" in out["permissionDecisionReason"]


@pytest.mark.parametrize(
    "payload",
    [
        _codex("Bash", {"command": "ls experiments"}),
        _codex("apply_patch", {"command": f"*** Update File: notes.md\n+see {STORE_FILE}\n"}),
        _codex("mcp__claude_ai_Slack__slack_read_channel", {"channel": "x"}),
    ],
    ids=["shell", "patch-that-mentions-the-store", "connector-read"],
)
def test_codex_allows(payload, repo: Path, monkeypatch, capsys) -> None:
    assert _reply("codex", payload, repo, monkeypatch, capsys) == ""


@pytest.mark.parametrize(
    ("payload", "permission"),
    [
        ({"hook_event_name": "beforeShellExecution", "command": f"head {STORE_FILE}"}, "deny"),
        ({"hook_event_name": "beforeShellExecution", "command": "ls"}, "allow"),
        ({"hook_event_name": "beforeReadFile", "file_path": STORE_FILE, "content": ""}, "deny"),
        (
            {"hook_event_name": "beforeReadFile", "file_path": "a.md", "content": STORE_FILE},
            "allow",
        ),
        ({"hook_event_name": "beforeMCPExecution", "tool_name": "slack_send_message"}, "deny"),
        ({"hook_event_name": "beforeMCPExecution", "tool_name": "slack_read_channel"}, "allow"),
    ],
)
def test_cursor_answers_every_call(payload, permission, repo, monkeypatch, capsys) -> None:
    assert (
        json.loads(_reply("cursor", payload, repo, monkeypatch, capsys))["permission"] == permission
    )


def test_a_repo_with_no_declarations_allows_connector_calls(tmp_path, monkeypatch, capsys) -> None:
    payload = _codex("mcp__claude_ai_Slack__slack_send_message", {})
    assert _reply("codex", payload, tmp_path, monkeypatch, capsys) == ""


def test_a_broken_guard_blocks_the_call(repo: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert guard.main(["codex", str(repo)]) == 2
    assert "blocked" in capsys.readouterr().err


def test_setup_writes_the_guard_for_codex_and_cursor(tmp_path: Path) -> None:
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    for hook in GUARD_HOOKS:
        entries = json.loads((tmp_path / hook.config_path).read_text())["hooks"][hook.event]
        assert [entry for entry in entries if guard.__name__ in json.dumps(entry)] == [hook.entry()]


def test_setup_keeps_a_repo_script_that_runs_the_packaged_hook(tmp_path: Path) -> None:
    script = tmp_path / "scripts/edit_hook.sh"
    script.parent.mkdir()
    script.write_text(f'#!/bin/sh\nexec sh .venv/lib/model_finder/hooks/{HOOK_SCRIPT} "$1"\n')
    link = tmp_path / ".agents/hooks" / HOOK_SCRIPT
    link.parent.mkdir(parents=True)
    link.symlink_to("../../scripts/edit_hook.sh")
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    assert link.readlink() == Path("../../scripts/edit_hook.sh")
