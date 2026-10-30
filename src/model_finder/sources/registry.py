"""Declarative source bundles + a tiny enable/list registry.

Each warehouse ships a bundle directory under ``bundles/<name>/`` with a
``source.json`` (how to reach it, the required read-only grant, auth modes — all
by reference, never a credential) and a ``SETUP.md``. Enabling a source records
intent in a local, gitignored user-config file and points at its SETUP.md.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from model_finder.io import load_json, write_json

BUNDLES_DIR = Path(__file__).resolve().parent / "bundles"
_REQUIRED_FIELDS = ("source_type", "display_name", "extra", "package", "driver", "description")
# Sources with their own adapter class in adapters.py. All others use ADBC through adbc_module.
NATIVE_ADAPTERS = frozenset({"databricks", "redshift", "duckdb"})
_CONFIG_PATH = Path.home() / ".config" / "model-finder" / "sources.json"
_OLD_CONFIG_PATH = Path.home() / ".config" / "agentic-ml-loop" / "sources.json"


class _EnabledSources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: list[str]


@dataclass(frozen=True)
class Bundle:
    source_type: str
    display_name: str
    extra: str
    package: str
    driver: str
    description: str
    directory: Path

    @property
    def setup_path(self) -> Path:
        return self.directory / "SETUP.md"


def _read_definition(path: Path) -> dict[str, Any] | None:
    """Read one source.json. A bad bundle is reported by `sources check`, and breaks nothing else."""
    try:
        definition = load_json(path)
    except ValueError:
        return None
    if not all(isinstance(definition.get(field), str) for field in _REQUIRED_FIELDS):
        return None
    return definition


def bundle_definitions() -> list[dict[str, Any]]:
    """Return each usable bundle's source.json. The adapters build their tables from these."""
    found = (_read_definition(path) for path in sorted(BUNDLES_DIR.glob("*/source.json")))
    return [definition for definition in found if definition is not None]


def _declared_extras() -> set[str]:
    try:
        declared = importlib.metadata.metadata("model-finder").get_all("Provides-Extra")
    except importlib.metadata.PackageNotFoundError:
        return set()
    return set(declared or [])


def bundle_problems() -> list[str]:
    """Say plainly what each bundle still needs before a pull can work."""
    import sqlglot

    problems: list[str] = []
    extras = _declared_extras()
    for path in sorted(BUNDLES_DIR.glob("*/source.json")):
        definition = _read_definition(path)
        if definition is None:
            problems.append(
                f"{path}: not valid JSON, or a required field is missing. "
                f"Fix: write valid JSON with: {', '.join(_REQUIRED_FIELDS)}"
            )
            continue
        name = definition["source_type"]
        try:
            sqlglot.Dialect.get_or_raise(name)
        except ValueError:
            problems.append(
                f"{path}: {name} is not a sqlglot dialect, so the read-only guard cannot parse "
                "its queries. Fix: name source_type after a sqlglot dialect"
            )
        if extras and definition["extra"] not in extras:
            problems.append(
                f"{path}: extra {definition['extra']} is not in pyproject.toml. "
                f"Fix: add it under [project.optional-dependencies] with {definition['package']}"
            )
        if "adbc_module" not in definition and name not in NATIVE_ADAPTERS:
            problems.append(
                f"{path}: {name} has no adapter. "
                "Fix: set adbc_module, or add an adapter class in sources/adapters.py"
            )
    return problems


def list_bundles() -> list[Bundle]:
    """Return all usable source bundles, sorted by source type."""
    bundles: list[Bundle] = []
    for path in sorted(BUNDLES_DIR.glob("*/source.json")):
        definition = _read_definition(path)
        if definition is not None:
            bundles.append(
                Bundle(
                    source_type=definition["source_type"],
                    display_name=definition["display_name"],
                    extra=definition["extra"],
                    package=definition["package"],
                    driver=definition["driver"],
                    description=definition["description"],
                    directory=path.parent,
                )
            )
    return bundles


def get_bundle(source_type: str) -> Bundle:
    for bundle in list_bundles():
        if bundle.source_type == source_type:
            return bundle
    raise KeyError(f"No source bundle named {source_type!r}")


def enabled_sources() -> set[str]:
    """Read enabled sources, and keep what a person enabled under the engine's old name."""
    enabled: set[str] = set()
    for path in (_OLD_CONFIG_PATH, _CONFIG_PATH):
        if path.exists():
            enabled |= set(_EnabledSources.model_validate(load_json(path)).enabled)
    return enabled


def is_enabled(source_type: str) -> bool:
    return source_type in enabled_sources()


def enable_source(source_type: str) -> None:
    get_bundle(source_type)  # validate it exists
    enabled = enabled_sources()
    enabled.add(source_type)
    write_json(_CONFIG_PATH, {"enabled": sorted(enabled)})
