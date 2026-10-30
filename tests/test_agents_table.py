"""Every name the Enforced rules table cites in AGENTS.md exists, so the table cannot drift."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCES = ("src", "scripts", "rules", "pyproject.toml", "Makefile")


def _cited_names() -> list[str]:
    table = (ROOT / "AGENTS.md").read_text().split("## Enforced rules", 1)[1]
    rows = [line.split("|") for line in table.splitlines() if line.startswith("| ")]
    enforced_by = [cells[2] for cells in rows[2:]]
    return [name for cell in enforced_by for name in re.findall(r"`([^`]+)`", cell)]


def _source_text() -> str:
    files = [path for name in SOURCES for path in [ROOT / name, *(ROOT / name).rglob("*")]]
    return "\n".join(
        path.read_text(errors="ignore")
        for path in files
        if path.is_file() and path.suffix != ".pyc"
    )


def test_every_cited_enforcement_exists() -> None:
    names = _cited_names()
    assert names, (
        "AGENTS.md: the Enforced rules table cites nothing. Fix: name each check in backticks"
    )
    text = _source_text()
    missing = [name for name in names if not (ROOT / name).exists() and name not in text]
    assert not missing, (
        f"AGENTS.md: the Enforced rules table cites {missing}, which no file defines. "
        "Fix: name the real file, test, function, or config key, or remove the row"
    )
