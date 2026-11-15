"""Instruction text stays small. Generate reference from code, and keep judgement in prose."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
# Current lines plus about 10%, so that a change has room to document itself.
LIMIT = 1310
INSTRUCTION_ROOTS = (
    ".agents",
    "demos",
    "examples",
    "src/model_finder/contract",
    "src/model_finder/skills",
    "src/model_finder/sources/bundles",
    "src/model_finder/templates",
)


def _instruction_files() -> list[Path]:
    found = {ROOT / name for name in ("AGENTS.md", "README.md", "SECURITY.md", "learnings.md")}
    for name in INSTRUCTION_ROOTS:
        found.update(path.resolve() for path in (ROOT / name).rglob("*.md"))
    return sorted(found)


def test_instruction_text_stays_under_the_limit() -> None:
    lines = sum(len(path.read_text().splitlines()) for path in _instruction_files())
    assert lines <= LIMIT, (
        f"instruction text is {lines} lines, over {LIMIT}. "
        "Fix: generate it from code, or point at `experiment contract` and `experiment next`."
    )


def test_the_count_sees_known_instruction_files() -> None:
    """A count over no files passes. Prove it still reads the root and a shipped skill."""
    found = {str(path.relative_to(ROOT)) for path in _instruction_files()}
    assert {"AGENTS.md", "src/model_finder/skills/research/SKILL.md"} <= found
