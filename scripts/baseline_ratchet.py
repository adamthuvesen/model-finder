"""Fail when the basedpyright baseline holds more errors than on origin/main."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASELINE = ".basedpyright/baseline.json"


def _count(text: str) -> int:
    return sum(len(errors) for errors in json.loads(text)["files"].values())


def main() -> int:
    on_main = subprocess.run(
        ["git", "show", f"origin/main:{BASELINE}"], capture_output=True, text=True, check=False
    )
    if on_main.returncode != 0:
        return 0  # main has no baseline yet, so this branch may introduce the first one
    here, there = _count(Path(BASELINE).read_text()), _count(on_main.stdout)
    if here > there:
        print(
            f"{BASELINE}: {here} errors, {there} on origin/main. "
            "Fix: remove the new type errors instead of baselining them.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
