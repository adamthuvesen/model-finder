from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(f"{path}: {exc.msg}", exc.doc, exc.pos) from exc


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable value, rejecting non-finite numbers."""
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return {str(k): _json_safe(v) for k, v in mapping.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in cast("list[object]", value)]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Cannot write non-finite JSON number: {value!r}")
        return value
    return value


def json_text(payload: Any) -> str:
    """Return the exact text ``write_json`` writes, so a hash can precede the write."""
    return json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json_text(payload)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        Path(tmp_name).replace(path)
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
