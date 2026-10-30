"""What agents may not reach: the holdout store, and a repo's declared connector write tools.

`experiment setup` turns these into Claude Code deny rules, and the guard hook enforces them for Codex
and Cursor. Both read them here, so there is one list. This module stays light: hooks import it on
every tool call.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Annotated, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Every holdout store path holds this, whatever the data home.
HOLDOUT_STORE_MARKER = "model-finder/holdouts"
# No machine path, so setup writes the same bytes everywhere and any data home is covered.
HOLDOUT_DENY_RULES = (
    f"Read(//**/{HOLDOUT_STORE_MARKER}/**)",
    f"Edit(//**/{HOLDOUT_STORE_MARKER}/**)",
    f"Bash(*{HOLDOUT_STORE_MARKER}*)",
)
# The forms earlier versions wrote, which named one machine's store. Setup replaces them.
EARLIER_DENY_RULE = re.compile(r"(Read|Edit)\((//|~/).*model-finder/holdouts/\*\*\)")
# A repo that declares its connectors hands every rule with this prefix to setup.
CONNECTOR_RULE_PREFIX = "mcp__claude_ai_"


class Connector(BaseModel):
    """A claude.ai connector, and the tools of it that write. Agents may not call those tools."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_]+$")]
    write_tools: Annotated[
        list[Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+$")]], Field(min_length=1)
    ]


class ToolSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connectors: list[Connector]


def declared_connectors(repo_root: Path) -> list[Connector] | None:
    """The repo's `[[tool.model-finder.connectors]]`. None when the repo declares none."""
    path = repo_root / "pyproject.toml"
    if not path.exists():
        return None
    try:
        tool = cast("dict[str, object]", tomllib.loads(path.read_text()).get("tool", {}))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: {exc}. Fix: repair the TOML, then run setup") from exc
    if "model-finder" not in tool:
        return None
    try:
        return ToolSettings.model_validate(tool["model-finder"]).connectors
    except ValidationError as exc:
        problem = exc.errors()[0]
        where = ".".join(str(part) for part in problem["loc"])
        raise ValueError(
            f"{path}: [tool.model-finder] {where}: {problem['msg']}. Fix: give each "
            "[[tool.model-finder.connectors]] a name and a list of write_tools"
        ) from exc


def connector_rules(repo_root: Path) -> list[str] | None:
    """The Claude Code deny rules for the declared write tools. None when the repo declares none."""
    connectors = declared_connectors(repo_root)
    if connectors is None:
        return None
    rules = (
        f"{CONNECTOR_RULE_PREFIX}{connector.name}__{write_tool}"
        for connector in connectors
        for write_tool in connector.write_tools
    )
    return list(dict.fromkeys(rules))
