"""`experiment setup`: link the packaged skills and deny the holdout store in a consumer repository."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

from agentic_ml_loop.io import read_text, write_json
from agentic_ml_loop.ledger import store_root

PACKAGED_SKILLS = Path(__file__).parents[1] / "skills"
SKILLS_DIR = (".agents", "skills")
ENGINE_BASH_RULE = "Bash(*agentic-ml-loop/holdouts*)"
# Only the rules this command writes match, so a rule from an older data home can be replaced.
ENGINE_PATH_RULE = re.compile(r"(Read|Edit)\((//|~/).*agentic-ml-loop/holdouts/\*\*\)")


def _is_engine_link(link: Path) -> bool:
    """Recognize a link to an older install of this package, such as a rebuilt venv."""
    return link.readlink().as_posix().endswith(f"agentic_ml_loop/skills/{link.name}")


def _link_skills(repo_root: Path) -> list[str]:
    """Link each packaged skill into `.agents/skills`, leaving a local skill alone."""
    skills_dir = repo_root.joinpath(*SKILLS_DIR)
    skills_dir.mkdir(parents=True, exist_ok=True)
    actions: list[str] = []
    for source in sorted(path for path in PACKAGED_SKILLS.iterdir() if path.is_dir()):
        link = skills_dir / source.name
        if link.is_symlink() and link.resolve() == source.resolve():
            continue
        if link.is_symlink() and _is_engine_link(link):
            link.unlink()
        elif link.is_symlink() or link.exists():
            actions.append(f"kept local skill {source.name}")
            continue
        link.symlink_to(source.resolve().relative_to(skills_dir.resolve(), walk_up=True))
        actions.append(f"linked skill {source.name} into .agents/skills")
    return actions


def _link_claude_skills(repo_root: Path) -> list[str]:
    """Point `.claude/skills` at `.agents/skills`, the path Claude Code reads."""
    link = repo_root / ".claude" / "skills"
    if link.exists() or link.is_symlink():
        return []
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(Path("..").joinpath(*SKILLS_DIR))
    return ["linked .claude/skills to .agents/skills"]


def _holdout_deny_rules() -> list[str]:
    """Name the store as `~/` when it can, so a committed file works for every user."""
    store = store_root().resolve()
    home = Path.home().resolve()
    pattern = (
        f"~/{store.relative_to(home).as_posix()}"
        if store.is_relative_to(home)
        else f"/{store.as_posix()}"
    )
    return [f"Read({pattern}/**)", f"Edit({pattern}/**)", ENGINE_BASH_RULE]


def _member(parent: object, key: str, default: object) -> object:
    if not isinstance(parent, dict):
        return None
    return cast("dict[str, object]", parent).setdefault(key, default)


def _deny_holdout_store(repo_root: Path) -> list[str]:
    """Add the deny rules to `.claude/settings.json`. Every other setting stays as it is."""
    path = repo_root / ".claude" / "settings.json"
    try:
        settings: object = json.loads(read_text(path)) if path.exists() else {}
    except ValueError as exc:
        raise ValueError(f"Cannot add holdout deny rules to {path}: {exc}") from exc
    deny = _member(_member(settings, "permissions", {}), "deny", [])
    if not isinstance(deny, list):
        raise TypeError(
            f"Cannot add holdout deny rules to {path}: permissions.deny must be a list in an object"
        )
    rules = cast("list[str]", deny)
    wanted = _holdout_deny_rules()
    kept = [rule for rule in rules if rule in wanted or not ENGINE_PATH_RULE.fullmatch(rule)]
    merged = kept + [rule for rule in wanted if rule not in kept]
    if merged == rules:
        return []
    rules[:] = merged
    write_json(path, settings)
    return ["wrote holdout deny rules to .claude/settings.json"]


def setup_repo(repo_root: Path) -> list[str]:
    return [
        *_deny_holdout_store(repo_root),
        *_link_skills(repo_root),
        *_link_claude_skills(repo_root),
    ]
