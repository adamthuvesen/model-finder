"""`experiment setup`: give a repository the skills, the holdout deny rule, and the agent hooks."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from model_finder.io import read_text, write_json

PACKAGED_SKILLS = Path(__file__).parents[1] / "skills"
SKILLS_DIR = (".agents", "skills")
OLD_ENGINE_NAMES = ("agentic-ml-loop",)
OLD_PACKAGES = ("agentic_ml_loop",)


def _deny_rules_for(name: str) -> tuple[str, str, str]:
    return (
        f"Read(//**/{name}/holdouts/**)",
        f"Edit(//**/{name}/holdouts/**)",
        f"Bash(*{name}/holdouts*)",
    )


# No machine path, so setup writes the same bytes everywhere and any data home is covered.
# The old name stays denied, because a holdout can still sit in a store from that name.
HOLDOUT_DENY_RULES = tuple(
    rule for name in ("model-finder", *OLD_ENGINE_NAMES) for rule in _deny_rules_for(name)
)
# The forms earlier versions wrote, which named one machine's store. Setup replaces them.
EARLIER_DENY_RULE = re.compile(
    r"(Read|Edit)\((//|~/).*(model-finder|agentic-ml-loop)/holdouts/\*\*\)"
)


HOOK_SCRIPT = "post_edit.sh"


def _hook_command(client: str) -> str:
    return f'sh "$(git rev-parse --show-toplevel)/.agents/hooks/{HOOK_SCRIPT}" {client}'


@dataclass(frozen=True)
class HookClient:
    """One client's documented hook format. Every hook comes from this table."""

    name: str
    config_path: str
    event: str
    matcher: str
    nested: bool = True
    fields: Mapping[str, object] = field(default_factory=dict[str, object])

    def entry(self) -> dict[str, object]:
        command = _hook_command(self.name)
        if self.nested:
            return {
                "matcher": self.matcher,
                "hooks": [{"type": "command", "command": command}],
            }
        return {"matcher": self.matcher, "command": command}


# Each shape follows the client's published docs. None was run against a live client.
# Cursor documents `additional_context` on `postToolUse`, and no output on `afterFileEdit`.
HOOK_CLIENTS = (
    HookClient("claude", ".claude/settings.json", "PostToolUse", "Edit|Write"),
    HookClient("codex", ".codex/hooks.json", "PostToolUse", "Edit|Write"),
    HookClient(
        "cursor", ".cursor/hooks.json", "postToolUse", "Write", nested=False, fields={"version": 1}
    ),
)


def _is_engine_link(link: Path) -> bool:
    """Recognize a link to an older install of this package, such as a rebuilt venv."""
    target = link.readlink().as_posix()
    return any(
        target.endswith(f"{package}/skills/{link.name}")
        for package in ("model_finder", *OLD_PACKAGES)
    )


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
        raise ValueError(
            f"{path}: Cannot add holdout deny rules: {exc}. Fix: repair the JSON, then run setup"
        ) from exc
    deny = _member(_member(settings, "permissions", {}), "deny", [])
    if not isinstance(deny, list):
        raise TypeError(
            f"{path}: Cannot add holdout deny rules: permissions.deny must be a list of strings "
            "in an object. Fix: repair that key, then run setup"
        )
    for index, rule in enumerate(cast("list[object]", deny)):
        if not isinstance(rule, str):
            raise TypeError(
                f"{path}: Cannot add holdout deny rules: permissions.deny[{index}] must be "
                "a string. Fix: write that deny rule as a string, then run setup"
            )
    rules = cast("list[str]", deny)
    kept = [
        rule
        for rule in rules
        if rule in HOLDOUT_DENY_RULES or not EARLIER_DENY_RULE.fullmatch(rule)
    ]
    merged = kept + [rule for rule in HOLDOUT_DENY_RULES if rule not in kept]
    if merged == rules:
        return []
    rules[:] = merged
    write_json(path, settings)
    return ["wrote holdout deny rules to .claude/settings.json"]


def _load_settings(path: Path) -> dict[str, object]:
    try:
        loaded: object = json.loads(read_text(path)) if path.exists() else {}
    except ValueError as exc:
        raise ValueError(
            f"{path}: Cannot add hooks: {exc}. Fix: repair the JSON, then run setup"
        ) from exc
    if not isinstance(loaded, dict):
        raise TypeError(f"{path}: Cannot add hooks: not an object. Fix: repair it, then run setup")
    return cast("dict[str, object]", loaded)


def _install_hook(repo_root: Path, client: HookClient) -> list[str]:
    """Replace this engine's hook in one client's config. Other hooks stay as they are."""
    path = repo_root / client.config_path
    settings = _load_settings(path)
    before = json.dumps(settings)
    for key, value in client.fields.items():
        settings.setdefault(key, value)
    hooks = _member(settings, "hooks", {})
    entries = _member(hooks, client.event, [])
    if not isinstance(entries, list):
        raise TypeError(
            f"{path}: hooks.{client.event} must be a list. Fix: repair it, then run setup"
        )
    others = [
        entry for entry in cast("list[object]", entries) if HOOK_SCRIPT not in json.dumps(entry)
    ]
    entries[:] = [*others, client.entry()]
    if json.dumps(settings) == before:
        return []
    write_json(path, settings)
    return [f"wrote the post-edit hook to {client.config_path}"]


def _link_hook_script(repo_root: Path) -> list[str]:
    link = repo_root / ".agents" / "hooks" / HOOK_SCRIPT
    source = (Path(__file__).parents[1] / "hooks" / HOOK_SCRIPT).resolve()
    if link.is_symlink() and link.resolve() == source:
        return []
    link.parent.mkdir(parents=True, exist_ok=True)
    link.unlink(missing_ok=True)
    link.symlink_to(source.relative_to(link.parent.resolve(), walk_up=True))
    return [f"linked {link.relative_to(repo_root)}"]


def setup_repo(repo_root: Path) -> list[str]:
    actions = [
        *_deny_holdout_store(repo_root),
        *_link_skills(repo_root),
        *_link_claude_skills(repo_root),
        *_link_hook_script(repo_root),
    ]
    for client in HOOK_CLIENTS:
        actions.extend(_install_hook(repo_root, client))
    return actions
