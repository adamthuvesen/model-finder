"""Check only the Python files that changed. `make check` is the full gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BIN = Path(".venv/bin")


def _changed_python_files() -> list[str]:
    listings = (
        ["git", "diff", "--name-only", "-z", "HEAD"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    )
    names: set[str] = set()
    for command in listings:
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        names.update(name for name in done.stdout.split("\0") if name)
    return sorted(name for name in names if name.endswith(".py") and Path(name).is_file())


def _tool_commands(files: list[str]) -> list[list[str]]:
    commands = [
        [
            str(BIN / "ruff"),
            "check",
            "--force-exclude",
            "--quiet",
            "--output-format=concise",
            *files,
        ],
        [str(BIN / "ruff"), "format", "--force-exclude", "--check", "--quiet", *files],
        [str(BIN / "basedpyright"), "--level", "error", *files],
    ]
    if Path("sgconfig.yml").is_file():
        commands.append([str(BIN / "ast-grep"), "scan", "--error", *files])
    return commands


def main(arguments: list[str]) -> int:
    files = arguments or _changed_python_files()
    if not files:
        print("quick: no changed Python files")
        return 0
    failed = False
    for command in _tool_commands(files):
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        if done.returncode:
            failed = True
            print(done.stdout + done.stderr, end="")
    if not failed:
        print(f"quick: ok ({len(files)} files)")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
