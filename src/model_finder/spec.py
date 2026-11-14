"""Read the machine-parsed fields of ``experiment.md`` and seed new experiments."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from model_finder.candidate_result import Direction
from model_finder.io import read_text
from model_finder.paths import SPECIFICATION_FILE_NAME

DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "templates" / "model-search.md"
_OBJECTIVE_SECTION = re.compile(
    r"^## Objective Metric\s*\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL
)
_PARENTHESIZED_TOKEN = re.compile(r"\(`?([a-z][a-z0-9_]*)`?\)")


def get_objective_metric(experiment_dir: Path) -> str | None:
    """Return the ``(val_auc)`` style token in the ``## Objective Metric`` section."""
    spec_path = experiment_dir / SPECIFICATION_FILE_NAME
    if not spec_path.exists():
        return None
    section = _OBJECTIVE_SECTION.search(read_text(spec_path))
    token = _PARENTHESIZED_TOKEN.search(section.group(1)) if section else None
    return token.group(1) if token else None


def get_objective_direction(experiment_dir: Path) -> Direction:
    """Experiments maximize unless ``objective_direction: minimize`` overrides it."""
    spec_path = experiment_dir / SPECIFICATION_FILE_NAME
    if not spec_path.exists():
        return "maximize"
    declared = re.findall(r"^objective_direction:[ \t]*(.*)$", read_text(spec_path), re.MULTILINE)
    directions = {value.strip().strip("`") for value in declared}
    if not directions:
        return "maximize"
    if len(directions) != 1:
        raise ValueError(
            f"{spec_path}: Conflicting objective_direction declarations. Fix: keep one line"
        )
    direction = directions.pop()
    if direction not in {"minimize", "maximize"}:
        raise ValueError(
            f"{spec_path}: objective_direction must be minimize or maximize. "
            "Fix: write `objective_direction: minimize` or `objective_direction: maximize`"
        )
    return cast("Direction", direction)


def get_key_columns(experiment_dir: Path) -> list[str]:
    """Return columns from a ``key_columns: a, b`` line, for the overlap report."""
    spec_path = experiment_dir / SPECIFICATION_FILE_NAME
    if not spec_path.exists():
        return []
    declared = re.search(r"^key_columns:[ \t]*(.*)$", read_text(spec_path), re.MULTILINE)
    if declared is None:
        return []
    columns = [column.strip().strip("`") for column in declared.group(1).split(",")]
    return list(dict.fromkeys(column for column in columns if column))


def research_journal_template(experiment_id: str) -> str:
    return (
        f"# Research Journal: {experiment_id}\n\n"
        "Head each entry `## Cycle NNNN: <title>`.\n"
        "Write one entry per hypothesis: what you set out to learn, what you found,\n"
        "and what it means for next steps.\n"
    )


def research_sources_template(experiment_id: str) -> str:
    return f"""# Research Sources: {experiment_id}

Durable, source-backed research memory for this experiment.
Keep hypotheses and results in `research_journal.md`.

## Reusable Takeaways

Rewrite these bullets when later findings narrow or contradict them.
This section is the current synthesis. It wins over a source card.

- _none yet_

## Source Cards

### Source 001: <title>

- **Type:** kaggle | paper | blog | docs
- **URL:** <link>
- **Why relevant here:** <one or two sentences>
- **Key takeaways:** <short bullets or prose>
- **Applicability / caveats:** <where this helps or does not apply>
- **Status:** used | deferred | ruled_out
"""
