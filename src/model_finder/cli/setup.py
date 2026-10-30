"""`experiment setup`: give a repository the skills, the holdout deny rule, and the agent hooks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from model_finder.cli.agent_rules import (
    CONNECTOR_RULE_PREFIX,
    EARLIER_DENY_RULE,
    HOLDOUT_DENY_RULES,
    connector_rules,
)
from model_finder.io import read_text, write_json

PACKAGED_SKILLS = Path(__file__).parents[1] / "skills"
SKILLS_DIR = (".agents", "skills")


HOOK_SCRIPT = "post_edit.sh"
GUARD_MODULE = "model_finder.cli.guard"
_TOP = "$(git rev-parse --show-toplevel)"


def _hook_command(owner: str, client: str) -> str:
    if owner == GUARD_MODULE:
        return f'"{_TOP}/.venv/bin/python" -m {GUARD_MODULE} {client} "{_TOP}"'
    return f'sh "{_TOP}/.agents/hooks/{HOOK_SCRIPT}" {client}'


@dataclass(frozen=True)
class HookClient:
    """One hook in one client's documented format. Every hook comes from these tables."""

    name: str
    config_path: str
    event: str
    matcher: str | None
    nested: bool = True
    fields: Mapping[str, object] = field(default_factory=dict[str, object])
    owner: str = HOOK_SCRIPT
    options: Mapping[str, object] = field(default_factory=dict[str, object])

    def entry(self) -> dict[str, object]:
        command = {"command": _hook_command(self.owner, self.name), **self.options}
        matcher = {} if self.matcher is None else {"matcher": self.matcher}
        if self.nested:
            return {**matcher, "hooks": [{"type": "command", **command}]}
        return {**matcher, **command}


# Each shape follows the client's published docs. None was run against a live client.
# Cursor documents `additional_context` on `postToolUse`, and no output on `afterFileEdit`.
HOOK_CLIENTS = (
    HookClient("claude", ".claude/settings.json", "PostToolUse", "Edit|Write"),
    HookClient("codex", ".codex/hooks.json", "PostToolUse", "Edit|Write"),
    HookClient(
        "cursor", ".cursor/hooks.json", "postToolUse", "Write", nested=False, fields={"version": 1}
    ),
)
# Codex and Cursor have no deny rules, so a pre-tool guard enforces the same limits. Codex reads
# files through the shell. A guard that cannot run blocks the call: Cursor fails closed.
_GUARD_OPTIONS = {"timeout": 10}
_CURSOR_GUARD = {"timeout": 10, "failClosed": True}
GUARD_HOOKS = (
    HookClient(
        "codex",
        ".codex/hooks.json",
        "PreToolUse",
        "Bash|apply_patch|mcp__.*",
        owner=GUARD_MODULE,
        options=_GUARD_OPTIONS,
    ),
    *(
        HookClient(
            "cursor",
            ".cursor/hooks.json",
            event,
            None,
            nested=False,
            fields={"version": 1},
            owner=GUARD_MODULE,
            options=_CURSOR_GUARD,
        )
        for event in ("beforeShellExecution", "beforeReadFile", "beforeMCPExecution")
    ),
)


def _is_engine_link(link: Path) -> bool:
    """Recognize a link to an older install of this package, such as a rebuilt venv."""
    return link.readlink().as_posix().endswith(f"model_finder/skills/{link.name}")


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


def _write_deny_rules(repo_root: Path) -> list[str]:
    """Write the holdout and connector deny rules. Every other setting stays as it is."""
    declared_rules = connector_rules(repo_root)
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
        if (rule in HOLDOUT_DENY_RULES or not EARLIER_DENY_RULE.fullmatch(rule))
        and (declared_rules is None or not rule.startswith(CONNECTOR_RULE_PREFIX))
    ]
    merged = [
        *kept,
        *(rule for rule in HOLDOUT_DENY_RULES if rule not in kept),
        *(declared_rules or []),
    ]
    if merged == rules:
        return []
    rules[:] = merged
    write_json(path, settings)
    return ["wrote holdout and connector deny rules to .claude/settings.json"]


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
        entry for entry in cast("list[object]", entries) if client.owner not in json.dumps(entry)
    ]
    entries[:] = [*others, client.entry()]
    if json.dumps(settings) == before:
        return []
    write_json(path, settings)
    kind = "guard" if client.owner == GUARD_MODULE else "post-edit"
    return [f"wrote the {kind} hook for {client.event} to {client.config_path}"]


def _link_hook_script(repo_root: Path) -> list[str]:
    link = repo_root / ".agents" / "hooks" / HOOK_SCRIPT
    source = (Path(__file__).parents[1] / "hooks" / HOOK_SCRIPT).resolve()
    if link.is_symlink() and link.resolve() == source:
        return []
    # A repo's own script that runs the packaged hook, so the hook also answers in a fresh clone.
    if link.is_file() and f"model_finder/hooks/{HOOK_SCRIPT}" in read_text(link.resolve()):
        return []
    link.parent.mkdir(parents=True, exist_ok=True)
    link.unlink(missing_ok=True)
    link.symlink_to(source.relative_to(link.parent.resolve(), walk_up=True))
    return [f"linked {link.relative_to(repo_root)}"]


def setup_repo(repo_root: Path) -> list[str]:
    actions = [
        *_write_deny_rules(repo_root),
        *_link_skills(repo_root),
        *_link_claude_skills(repo_root),
        *_link_hook_script(repo_root),
    ]
    for client in (*HOOK_CLIENTS, *GUARD_HOOKS):
        actions.extend(_install_hook(repo_root, client))
    return actions
