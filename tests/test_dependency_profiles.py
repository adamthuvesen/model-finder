from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _package_name(requirement: str) -> str:
    return requirement.split("[", 1)[0].split(">=", 1)[0].split("==", 1)[0]


def test_base_dependencies_exclude_every_optional_package() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = {_package_name(req) for req in pyproject["project"]["dependencies"]}
    optional = {
        _package_name(req)
        for requirements in pyproject["project"]["optional-dependencies"].values()
        for req in requirements
    }

    assert base.isdisjoint(optional), (
        f"pyproject.toml: {sorted(base & optional)} are base and optional dependencies. "
        "Fix: keep each in its extra only, so a plain install stays light"
    )
