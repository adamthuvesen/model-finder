"""Engine code reaches DuckDB only through the source adapters.

TID251 cannot hold this ban: it would also stop local CSV analysis in experiments/ and demos/,
and a per-file ignore would lift every other ban for the owner file.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "model_finder"
OWNERS = frozenset({ENGINE / "sources" / "adapters.py"})


def imports_duckdb(tree: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        if any(name == "duckdb" or name.startswith("duckdb.") for name in names):
            lines.append(node.lineno)
    return lines


def duckdb_imports(engine: Path, owners: frozenset[Path]) -> list[str]:
    return [
        f"{path.relative_to(engine)}:{line}"
        for path in sorted(engine.rglob("*.py"))
        if path not in owners
        for line in imports_duckdb(ast.parse(path.read_text(), filename=str(path)))
    ]


def test_engine_code_does_not_import_duckdb() -> None:
    found = duckdb_imports(ENGINE, OWNERS)
    assert not found, (
        "duckdb is imported outside the source adapters: "
        + ", ".join(found)
        + ". Fix: add an adapter in src/model_finder/sources/adapters.py."
    )


def test_the_scan_finds_every_import_form_and_skips_the_owner(tmp_path: Path) -> None:
    """A scan that finds nothing passes, so prove it still finds known cases."""
    (tmp_path / "plain.py").write_text("import duckdb\n")
    (tmp_path / "from_form.py").write_text("from duckdb import connect\n")
    (tmp_path / "lazy.py").write_text("def f():\n    import duckdb.typing\n")
    (tmp_path / "owner.py").write_text("import duckdb\n")
    (tmp_path / "fine.py").write_text("import duckdb_extras\nx = 'import duckdb'\n")

    found = duckdb_imports(tmp_path, frozenset({tmp_path / "owner.py"}))

    assert found == ["from_form.py:1", "lazy.py:2", "plain.py:1"]
