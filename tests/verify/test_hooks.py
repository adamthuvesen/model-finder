"""The post-edit hook reports findings for each client, and it never blocks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from model_finder.cli.hook import edited_files, experiment_of, report
from model_finder.cli.main import main
from model_finder.cli.setup import HOOK_CLIENTS

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "src/model_finder/hooks/post_edit.sh"


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"tool_input": {"file_path": "a.py"}}, ["a.py"]),
        ({"file_path": "b.py"}, ["b.py"]),
        ({"tool_input": {"path": "c.py"}}, ["c.py"]),
        (
            {"tool_input": {"command": "*** Update File: d.py\n*** Add File: e.py\n"}},
            ["d.py", "e.py"],
        ),
        ({"tool_input": {"command": "ls"}}, []),
        ({}, []),
    ],
)
def test_edited_files_reads_every_client(event, expected, tmp_path: Path) -> None:
    assert edited_files(event, tmp_path) == [str(tmp_path.resolve() / name) for name in expected]


def test_report_uses_each_clients_documented_shape() -> None:
    assert json.loads(report("cursor", "x")) == {"additional_context": "x"}
    for client in ("claude", "codex"):
        output = json.loads(report(client, "x"))["hookSpecificOutput"]
        assert output == {"hookEventName": "PostToolUse", "additionalContext": "x"}


def test_experiment_of_finds_the_experiment(tmp_path: Path) -> None:
    (tmp_path / "e").mkdir()
    (tmp_path / "e/experiment.md").write_text("x")
    assert experiment_of(tmp_path / "e/notes.md") == (tmp_path / "e").resolve()
    assert experiment_of(tmp_path / "other.md") is None


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    bin_dir = repo / ".venv/bin"
    bin_dir.mkdir(parents=True)
    for tool in ("python", "ruff", "basedpyright", "experiment"):
        wrapper = bin_dir / tool
        wrapper.write_text(f'#!/bin/sh\nexec "{Path(sys.executable).parent / tool}" "$@"\n')
        wrapper.chmod(0o755)
    return repo


def _run_hook(repo: Path, file: str, client: str = "claude") -> subprocess.CompletedProcess[str]:
    event = {"tool_input": {"file_path": file}}
    return subprocess.run(
        ["sh", str(SCRIPT), client],
        cwd=repo,
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": os.environ["PATH"]},
        check=False,
    )


def test_a_python_edit_is_formatted_and_its_findings_are_reported(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "bad.py").write_text("import os\nx=1\n")
    done = _run_hook(repo, "bad.py")
    assert done.returncode == 0
    context = json.loads(done.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "bad.py:1" in context
    assert (repo / "bad.py").read_text().endswith("x = 1\n")


def test_a_skill_edit_is_type_checked_as_the_skills_own_project(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    skill = repo / ".agents/skills/verify-demo"
    skill.mkdir(parents=True)
    (skill / "pyrightconfig.json").write_text('{"include": ["."]}')
    (skill / "helper.py").write_text('VALUE: int = "not an int"\n')
    done = _run_hook(repo, str(skill / "helper.py"))
    assert done.returncode == 0
    context = json.loads(done.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "helper.py:1:14 - error" in context


@pytest.mark.parametrize("has_type_config", [False, True])
def test_a_shell_verify_script_in_a_skill_says_nothing(
    tmp_path: Path, *, has_type_config: bool
) -> None:
    repo = _repo(tmp_path)
    skill = repo / ".agents/skills/verify-demo"
    skill.mkdir(parents=True)
    if has_type_config:
        (skill / "pyrightconfig.json").write_text('{"include": ["."]}')
    (skill / "verify").write_text(
        '#!/bin/bash\nset -euo pipefail\nif [ -n "$1" ]; then echo "$1"; fi\n'
    )
    done = _run_hook(repo, str(skill / "verify"))
    assert done.returncode == 0
    assert done.stdout == ""


def test_a_python_verify_script_in_a_skill_without_a_type_config_is_linted(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    skill = repo / ".agents/skills/verify-demo"
    skill.mkdir(parents=True)
    (skill / "verify").write_text("#!/usr/bin/env python3\nimport os\n")
    done = _run_hook(repo, str(skill / "verify"))
    context = json.loads(done.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "verify:2:8: F401" in context


def test_a_clean_edit_says_nothing(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "good.py").write_text("VALUE = 1\n")
    done = _run_hook(repo, "good.py")
    assert done.returncode == 0
    assert done.stdout == ""


def test_an_edit_to_the_leaderboard_gets_the_repair_command(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "results.json").write_text("[]")
    done = _run_hook(repo, "results.json", "cursor")
    assert done.returncode == 0
    assert "experiment rebuild-results" in json.loads(done.stdout)["additional_context"]


def test_setup_installs_one_hook_per_client_and_keeps_others(tmp_path: Path, capsys) -> None:
    settings = tmp_path / ".claude/settings.json"
    settings.parent.mkdir()
    theirs = {"matcher": "Bash", "hooks": [{"type": "command", "command": "theirs.sh"}]}
    settings.write_text(json.dumps({"hooks": {"PostToolUse": [theirs]}}))
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    entries = json.loads(settings.read_text())["hooks"]["PostToolUse"]
    assert entries[0] == theirs
    assert entries[1]["matcher"] == "Edit|Write"
    assert 'post_edit.sh" claude' in entries[1]["hooks"][0]["command"]
    codex = json.loads((tmp_path / ".codex/hooks.json").read_text())
    assert codex["hooks"]["PostToolUse"][0]["hooks"][0]["command"].endswith('post_edit.sh" codex')
    cursor = json.loads((tmp_path / ".cursor/hooks.json").read_text())
    assert cursor["version"] == 1
    assert cursor["hooks"]["postToolUse"][0]["command"].endswith('post_edit.sh" cursor')
    assert (tmp_path / ".agents/hooks/post_edit.sh").read_text().startswith("#!/bin/sh")
    capsys.readouterr()
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    assert capsys.readouterr().out == "nothing to do\n"


def test_every_client_comes_from_one_table() -> None:
    assert {client.name for client in HOOK_CLIENTS} == {"claude", "codex", "cursor"}


def test_the_hook_script_never_blocks_without_a_venv(tmp_path: Path) -> None:
    repo = tmp_path / "plain"
    repo.mkdir()
    assert shutil.which("git")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    assert _run_hook(repo, "a.py").returncode == 0
