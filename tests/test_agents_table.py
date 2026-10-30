"""Every check the Enforced Rules table in AGENTS.md cites is defined where it says."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
HEADING = "## Enforced Rules"


def _rows() -> list[tuple[str, list[str]]]:
    text = (ROOT / "AGENTS.md").read_text()
    assert f"\n{HEADING}\n" in text, f"AGENTS.md: no `{HEADING}` heading. Fix: restore it"
    table = text.split(f"\n{HEADING}\n", 1)[1].split("\n## ", 1)[0]
    rows = [line.split("|") for line in table.splitlines() if line.startswith("| ")][2:]
    assert rows, f"AGENTS.md: `{HEADING}` has no table rows. Fix: add the table"
    return [(cells[1].strip(), re.findall(r"`([^`]+)`", cells[2])) for cells in rows]


def _toml_defines(path: Path, dotted: str) -> bool:
    node: object = tomllib.loads(path.read_text())
    for key in dotted.split("."):
        if isinstance(node, list):
            node = next(
                (item for item in node if isinstance(item, dict) and item.get("id") == key), None
            )
        elif isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return False
    return node is not None


def _defines(path: Path, name: str) -> bool:
    if path.suffix == ".toml":
        return _toml_defines(path, name)
    escaped = re.escape(name)
    patterns = {
        ".py": rf"^\s*(?:async\s+)?(?:def|class)\s+{escaped}\b|^{escaped}\s*[:=]",
        ".yml": rf"^id:\s*{escaped}\s*$",
        "": rf"^{escaped}:",
    }
    pattern = patterns.get(path.suffix)
    return pattern is not None and re.search(pattern, path.read_text(), re.MULTILINE) is not None


def test_every_rule_cites_a_defined_check() -> None:
    problems: list[str] = []
    for rule, citations in _rows():
        if not citations:
            problems.append(f"{rule!r} cites nothing")
        for citation in citations:
            path, separator, name = citation.partition("::")
            if not separator or not (ROOT / path).is_file() or not _defines(ROOT / path, name):
                problems.append(f"{rule!r} cites `{citation}`, which is not defined there")
    assert not problems, (
        f"AGENTS.md: {'; '.join(problems)}. Fix: cite each check as `path::name`, where that "
        "file defines the name"
    )
