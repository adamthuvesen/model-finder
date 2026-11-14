"""No file may turn strict type checking off, or ignore a type error without a rule and a reason."""

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNGRADE = re.compile(
    r"^\s*#\s*pyright:\s*(?:[^#\n]*,\s*)?(?:basic|standard|off|recommended|all|strict\s*=\s*false)\b"
)
UNEXPLAINED_IGNORE = re.compile(r"#\s*pyright:\s*ignore(?!\[[^\]]+\]\s*--\s*\S)")


def is_python_file(path: Path) -> bool:
    if path.suffix:
        return path.suffix == ".py"
    with path.open("rb") as handle:
        first_line = handle.readline()
    return first_line.startswith(b"#!") and b"python" in first_line


def repo_python_files(root: Path) -> list[Path]:
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        env=env,
        capture_output=True,
        check=True,
    ).stdout.decode()
    paths = (root / name for name in listing.split("\0") if name)
    return [path for path in paths if path.is_file() and is_python_file(path)]


def downgrades(root: Path) -> list[str]:
    return [
        f"{path.relative_to(root)}:{number}"
        for path in repo_python_files(root)
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
        if DOWNGRADE.match(line) or UNEXPLAINED_IGNORE.search(line)
    ]


def test_no_file_downgrades_type_checking_mode() -> None:
    found = downgrades(ROOT)
    assert not found, (
        "a `# pyright:` comment turns strict mode off, or an ignore names no rule or reason: "
        + ", ".join(found)
    )


def test_the_scan_finds_a_downgrade_in_a_new_unstaged_file(tmp_path: Path) -> None:
    """A scan that finds nothing passes, so prove it still finds known cases."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, env=env, check=True)
    (tmp_path / "new.py").write_text("# pyright: basic\n")
    (tmp_path / "script").write_text("#!/usr/bin/env python3\n# pyright: strict=false\n")
    (tmp_path / "fine.py").write_text("# pyright: reportAny=false\n")
    hash_mark = "#"
    (tmp_path / "bare.py").write_text(f"x = 1  {hash_mark} pyright: ignore\n")
    (tmp_path / "reasonless.py").write_text(f"x = 1  {hash_mark} pyright: ignore[reportAny]\n")
    (tmp_path / "reasoned.py").write_text("x = 1  # pyright: ignore[reportAny] -- why\n")

    assert sorted(downgrades(tmp_path)) == [
        "bare.py:1",
        "new.py:1",
        "reasonless.py:1",
        "script:2",
    ]
