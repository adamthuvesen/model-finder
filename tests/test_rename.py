"""A store or a setup from the engine's old name stays safe, and it fails with an exact fix."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from model_finder.cli.errors import describe
from model_finder.cli.main import main
from model_finder.final_evaluation import evaluate_final, freeze_candidate
from model_finder.holdout import read_holdout_seal, sealed_file_path
from model_finder.ledger import entries_for, read_ledger, store_root
from model_finder.reseal import reseal_holdout
from tests.conftest import fitted, rows

RUNNERS = {"candidate-a": fitted}


def legacy_store_root() -> Path:
    return store_root().parent.parent / "agentic-ml-loop" / "holdouts"


OLD_NAME = "agentic-ml-loop"


def _move_store_to_the_old_name() -> Path:
    old = store_root().parent.parent / OLD_NAME / "holdouts"
    old.parent.mkdir(parents=True)
    shutil.move(store_root(), old)
    return old


def test_the_old_store_is_found_where_earlier_versions_wrote_it() -> None:
    assert legacy_store_root().parts[-2:] == (OLD_NAME, "holdouts")
    assert legacy_store_root().parent.parent == store_root().parent.parent


def test_a_holdout_in_the_old_store_fails_with_both_paths_and_an_mv(scored: Path) -> None:
    freeze_candidate(scored, "candidate-a", "label")
    seal = read_holdout_seal(scored)
    new_path = sealed_file_path(seal)
    old = _move_store_to_the_old_name()
    old_path = old / seal.experiment_uid / new_path.name
    assert old_path.is_file()
    with pytest.raises(FileNotFoundError) as caught:
        evaluate_final(scored, RUNNERS)
    message = describe(caught.value)
    assert str(old_path) in message
    assert str(new_path) in message
    assert "Fix: run `" in message
    assert old_path.is_file(), "the engine must not move a holdout on its own"
    assert not new_path.exists()

    command = re.search(r"Fix: run `(.+?)`", message)
    assert command
    subprocess.run(command.group(1), shell=True, check=True)  # noqa: S602
    assert evaluate_final(scored, RUNNERS).holdout == seal


def test_a_holdout_in_neither_store_keeps_its_own_message(scored: Path) -> None:
    seal = read_holdout_seal(scored)
    sealed_file_path(seal).unlink()
    freeze_candidate(scored, "candidate-a", "label")
    with pytest.raises(FileNotFoundError, match="missing from the store"):
        evaluate_final(scored, RUNNERS)


def test_a_read_recorded_in_the_old_ledger_still_counts(scored: Path) -> None:
    freeze_candidate(scored, "candidate-a", "label")
    evaluate_final(scored, RUNNERS)
    seal = read_holdout_seal(scored)
    keep = pd.read_parquet(sealed_file_path(seal))
    old = store_root().parent.parent / OLD_NAME / "holdouts"
    old.mkdir(parents=True)
    shutil.move(store_root() / "ledger.jsonl", old / "ledger.jsonl")
    assert entries_for(seal.holdout_sha256)[0].event == "evaluated"
    assert read_ledger()
    (scored / "work/selection.json").unlink()
    (scored / "work/final_evaluation_started.json").unlink()
    (scored / "outputs/final_evaluation.json").unlink()
    freeze_candidate(scored, "candidate-a", "label")
    with pytest.raises(ValueError, match="allow-reread"):
        evaluate_final(scored, RUNNERS)
    assert len(keep)


def test_a_corrupt_old_ledger_names_its_own_file(scored: Path) -> None:
    old = legacy_store_root()
    old.mkdir(parents=True)
    (old / "ledger.jsonl").write_text("not json\n")
    with pytest.raises(ValueError, match=rf"{old}/ledger\.jsonl:1"):
        read_ledger()


def test_setup_replaces_links_and_hooks_from_the_old_package(tmp_path: Path) -> None:
    old_package = "../../.venv/lib/python3.13/site-packages/agentic_ml_loop"
    skills = tmp_path / ".agents/skills"
    skills.mkdir(parents=True)
    (skills / "research").symlink_to(f"{old_package}/skills/research")
    hooks = tmp_path / ".agents/hooks"
    hooks.mkdir()
    (hooks / "post_edit.sh").symlink_to(f"{old_package}/hooks/post_edit.sh")
    settings = tmp_path / ".claude/settings.json"
    settings.parent.mkdir()
    old_hook = {
        "matcher": "Edit|Write",
        "hooks": [{"type": "command", "command": 'sh "x/.agents/hooks/post_edit.sh" claude'}],
    }
    theirs = {"matcher": "Bash", "hooks": [{"type": "command", "command": "theirs.sh"}]}
    settings.write_text(
        json.dumps(
            {
                "hooks": {"PostToolUse": [theirs, old_hook]},
                "permissions": {
                    "deny": [
                        "Read(./.env)",
                        "Read(~/.local/share/agentic-ml-loop/holdouts/**)",
                        "Edit(//Users/someone/agentic-ml-loop/holdouts/**)",
                        "Read(~/.local/share/model-finder/holdouts/**)",
                    ]
                },
            }
        )
    )
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    assert (skills / "research/SKILL.md").is_file()
    assert (hooks / "post_edit.sh").read_text().startswith("#!/bin/sh")
    loaded = json.loads(settings.read_text())
    commands = [entry["hooks"][0]["command"] for entry in loaded["hooks"]["PostToolUse"]]
    assert commands[0] == "theirs.sh"
    assert len(commands) == 2
    assert "x/.agents" not in commands[1]
    assert loaded["permissions"]["deny"] == [
        "Read(./.env)",
        "Read(//**/model-finder/holdouts/**)",
        "Edit(//**/model-finder/holdouts/**)",
        "Bash(*model-finder/holdouts*)",
        "Read(//**/agentic-ml-loop/holdouts/**)",
        "Edit(//**/agentic-ml-loop/holdouts/**)",
        "Bash(*agentic-ml-loop/holdouts*)",
    ]


def test_new_seals_go_to_the_new_store(scored: Path) -> None:
    seal = reseal_holdout(scored, rows(count=12, seed=99, start="2026-06-01"), "New rows.")
    assert "model-finder" in str(sealed_file_path(seal))
    assert OLD_NAME not in str(sealed_file_path(seal))
