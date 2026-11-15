"""No file may turn strict type checking off, or ignore a type error without a rule and a reason. No skip without a reason."""

import ast
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
        + ". Fix: remove the file-level comment, or write `# pyright: ignore[rule] -- reason` "
        "on the one line."
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


SKIP_NAMES = {"pytest.skip", "unittest.skip", "pytest.mark.skip", "pytest.mark.xfail"}
# Only these two take the reason as a first positional argument. For the marks, the first
# positional argument is the xfail condition, so they take the reason as a keyword alone.
POSITIONAL_REASON = {"pytest.skip", "unittest.skip"}


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _reason(call: ast.Call) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == "reason":
            return keyword.value
    if _dotted_name(call.func) in POSITIONAL_REASON and call.args:
        return call.args[0]
    return None


def _has_reason(value: ast.expr | None) -> bool:
    if value is None:
        return False
    return not (
        isinstance(value, ast.Constant) and isinstance(value.value, str) and not value.value.strip()
    )


def reasonless_skips(source: str) -> list[int]:
    found: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and _dotted_name(node.func) in SKIP_NAMES:
            if not _has_reason(_reason(node)):
                found.append(node.lineno)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found.extend(
                decorator.lineno
                for decorator in node.decorator_list
                if _dotted_name(decorator) in SKIP_NAMES
            )
    return sorted(found)


def test_every_skip_gives_a_reason() -> None:
    bare = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in repo_python_files(ROOT)
        for line in reasonless_skips(path.read_text(errors="replace"))
    ]
    assert not bare, (
        "give each skip or xfail a reason string: "
        + ", ".join(bare)
        + '. Fix: pass a reason, as in `@pytest.mark.skip(reason="needs the warehouse")`.'
    )


def test_skip_scan_finds_known_reasonless_skips() -> None:
    source = (
        "import pytest\n"
        "@pytest.mark.skip\n"
        "def test_bare_decorator(): ...\n"
        "@pytest.mark.skip()\n"
        "def test_empty_mark(): ...\n"
        "pytest.skip()\n"
        "pytest.skip('')\n"
        "pytest.mark.xfail(False)\n"
        "unittest.skip('the reason is positional')\n"
        "pytest.mark.skip(reason='the reason is a keyword')\n"
        "pytest.mark.xfail(False, reason='the reason is a keyword')\n"
        "pytest.skip(reason='the reason is a keyword')\n"
    )
    assert reasonless_skips(source) == [2, 4, 6, 7, 8]
