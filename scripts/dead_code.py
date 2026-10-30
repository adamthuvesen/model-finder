"""Run vulture with every Pydantic field counted as used, because the engine serializes them.

Vulture cannot see that a field is read by `model_dump`, so each new field would need a
hand-kept `ignore_names` entry. This script derives those names from the models instead.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

from pydantic import BaseModel
from vulture.config import make_config
from vulture.core import Vulture

import model_finder


def _subclasses(model: type[BaseModel]) -> list[type[BaseModel]]:
    return [found for sub in model.__subclasses__() for found in (sub, *_subclasses(sub))]


def _field_names() -> set[str]:
    for module in pkgutil.walk_packages(model_finder.__path__, "model_finder."):
        importlib.import_module(module.name)
    engine_models = [
        model for model in _subclasses(BaseModel) if model.__module__.startswith("model_finder.")
    ]
    return {"model_config", *(name for model in engine_models for name in model.model_fields)}


def main() -> int:
    config = make_config(sys.argv[1:])
    vulture = Vulture(
        ignore_names=config["ignore_names"], ignore_decorators=config["ignore_decorators"]
    )
    vulture.scavenge(config["paths"], exclude=config["exclude"])
    vulture.scan("\n".join(f"_.{name}" for name in sorted(_field_names())), "<pydantic fields>")
    found = vulture.report(min_confidence=config["min_confidence"])
    if found:
        print(
            "Fix: delete the code. If a caller vulture cannot see uses it, add the name to "
            "ignore_names under [tool.vulture] in pyproject.toml"
        )
    return int(found)


if __name__ == "__main__":
    raise SystemExit(main())
