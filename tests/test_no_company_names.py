"""The engine is public and company-neutral, so no company name enters src/ or tests/."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = (ROOT / "src", ROOT / "tests")
# Letters only on each side, so `mentions` passes and `menti_model_finder` fails.
COMPANY_NAME = re.compile(r"(?<![a-z])menti(?:meter)?(?![a-z])", re.IGNORECASE)


def company_names(roots: tuple[Path, ...], skip: Path) -> list[str]:
    return [
        f"{path.relative_to(ROOT)}:{number}"
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != skip and "__pycache__" not in path.parts
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1)
        if COMPANY_NAME.search(line)
    ]


def test_no_company_name_in_src_or_tests() -> None:
    found = company_names(SCANNED, Path(__file__).resolve())
    assert not found, (
        f"A company name is in the engine: {', '.join(found)}. "
        "Fix: move company-specific code to the consumer repo, or reword it."
    )


def test_the_pattern_finds_the_name_and_skips_ordinary_words() -> None:
    """A scan that finds nothing passes, so prove it still finds known cases."""
    hits = ["Mentimeter", "menti_model_finder", "MENTI"]
    misses = ["mentions", "implementing", "segmentation", "momentimeter"]
    assert all(COMPANY_NAME.search(text) for text in hits)
    assert not any(COMPANY_NAME.search(text) for text in misses)
