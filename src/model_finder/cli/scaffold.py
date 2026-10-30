"""`experiment new` and `experiment candidate new`: start from working code."""

from __future__ import annotations

import importlib.util
import json
import keyword
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

from model_finder.cli.runner import RunnerConfig, initialize_experiment
from model_finder.holdout import read_holdout_seal
from model_finder.io import write_text
from model_finder.ledger import store_root
from model_finder.paths import (
    HOLDOUT_SEAL_FILE_NAME,
    MODEL_DIRNAME,
    RUNNER_COMMAND_PATH,
    SPECIFICATION_FILE_NAME,
)
from model_finder.runner import save_candidate_result
from model_finder.splits import parsed_times

_TEMPLATES = Path(__file__).parents[1] / "templates"
_TARGET_NAMES = ("label", "target", "y", "outcome")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
FIRST_CANDIDATES = ("baseline", "linear")


def _template(name: str, **values: str) -> str:
    text = (_TEMPLATES / name).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace(f"__{key}__", value)
    return text


def _choose_target(rows: pd.DataFrame, target: str | None) -> str:
    named = [column for column in rows.columns if str(column).lower() in _TARGET_NAMES]
    chosen = target or (named[0] if named else str(rows.columns[-1]))
    if chosen not in rows.columns:
        raise ValueError(
            f"--target {chosen} is not a column. Fix: use one of: {', '.join(map(str, rows.columns))}"
        )
    if not pd.api.types.is_numeric_dtype(rows[chosen]):
        raise ValueError(
            f"target column {chosen} is not numeric. Fix: encode it as 0 and 1, or pass --target"
        )
    return chosen


def _parses_as_dates(rows: pd.DataFrame, column: str) -> bool:
    if pd.api.types.is_numeric_dtype(rows[column]):
        return False
    try:
        parsed_times(rows, column)
    except ValueError:
        return False
    return True


def _choose_time_column(rows: pd.DataFrame, time_column: str | None) -> str | None:
    if time_column is not None:
        if time_column not in rows.columns:
            raise ValueError(
                f"--time-column {time_column} is not a column. "
                f"Fix: use one of: {', '.join(map(str, rows.columns))}"
            )
        return time_column
    dates = [column for column in map(str, rows.columns) if _parses_as_dates(rows, column)]
    return dates[0] if dates else None


def load_runner(model_dir: Path) -> RunnerConfig:
    """Load a generated runner the way a script would, with its folder on the import path."""
    spec = importlib.util.spec_from_file_location("experiment_runner", model_dir / "run.py")
    if spec is None or spec.loader is None:
        raise ValueError(f"{model_dir / 'run.py'}: cannot be loaded. Fix: restore the file")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(model_dir))
    try:
        spec.loader.exec_module(module)
        return module.load_config()
    finally:
        sys.path.remove(str(model_dir))
        sys.modules.pop("candidates", None)


def _text(value: str | None) -> str:
    """Write a value as Python source in the formatter's own style."""
    return "None" if value is None else json.dumps(value)


def _write_code(
    experiment_dir: Path,
    root: Path,
    slug: str,
    csv: Path,
    target: str,
    time: str | None,
    *,
    binary: bool,
) -> None:
    model_dir = experiment_dir / MODEL_DIRNAME
    not_features = [target, *([time] if time else [])]
    write_text(
        experiment_dir / SPECIFICATION_FILE_NAME,
        _template(
            "scaffold_spec.md",
            SLUG=slug,
            METRIC="val_auc" if binary else "val_rmse",
            METRIC_NAME="Validation AUC" if binary else "Validation RMSE",
            DIRECTION="" if binary else "objective_direction: minimize",
        ),
    )
    write_text(
        model_dir / "run.py",
        _template(
            "scaffold_run.py.txt",
            SLUG_TEXT=_text(slug),
            SLUG=slug,
            RUN_PATH=(model_dir / "run.py").relative_to(root).as_posix(),
            CSV=_text(Path(os.path.relpath(csv.resolve(), model_dir.resolve())).as_posix()),
            TIME=_text(time),
        ),
    )
    write_text(
        model_dir / "candidates.py",
        _template(
            "scaffold_candidates.py.txt",
            SLUG=slug,
            TARGET=_text(target),
            NOT_FEATURES=json.dumps(not_features),
            KIND="classification" if binary else "regression",
        )
        + _template("scaffold_candidate.py.txt", NAME="linear"),
    )
    write_text(
        experiment_dir / RUNNER_COMMAND_PATH,
        f"python {(model_dir / 'run.py').relative_to(root).as_posix()}\n",
    )


def _remove_partial_experiment(experiment_dir: Path) -> None:
    """A failed `new` leaves nothing, so the same command works after the fix."""
    seal_path = experiment_dir / HOLDOUT_SEAL_FILE_NAME
    if seal_path.is_file():
        shutil.rmtree(
            store_root() / read_holdout_seal(experiment_dir).experiment_uid, ignore_errors=True
        )
    shutil.rmtree(experiment_dir, ignore_errors=True)
    if not any(experiment_dir.parent.iterdir()):
        experiment_dir.parent.rmdir()


def new_experiment(
    root: Path,
    slug: str,
    csv: Path,
    *,
    target: str | None = None,
    time_column: str | None = None,
    no_holdout: bool = False,
) -> Path:
    """Create an experiment with a runner and candidates, and run them on the person's rows."""
    if not _SLUG.match(slug):
        raise ValueError(
            f"slug {slug!r} is not a plain name. Fix: use letters, digits, `-`, and `_`"
        )
    experiment_dir = root / "experiments" / slug
    if experiment_dir.exists():
        raise ValueError(
            f"{experiment_dir}: already exists. Fix: choose another slug, or run "
            f"`experiment next {experiment_dir}`"
        )
    if not csv.is_file():
        raise FileNotFoundError(f"{csv}: no such file. Fix: pass the path to your CSV with --csv")
    rows = pd.read_csv(csv)
    chosen_target = _choose_target(rows, target)
    kept = rows.dropna(subset=[chosen_target])
    chosen_time = _choose_time_column(kept, time_column)
    binary = set(kept[chosen_target].unique()) <= {0, 1}
    finished = False
    try:
        _write_code(experiment_dir, root, slug, csv, chosen_target, chosen_time, binary=binary)
        config = load_runner(experiment_dir / MODEL_DIRNAME)
        print(
            initialize_experiment(
                config, experiment_dir, force=False, data_seed=None, no_holdout=no_holdout
            )
        )
        for candidate in FIRST_CANDIDATES:
            save_candidate_result(
                experiment_dir, candidate, config.candidate_runners, config.code_root
            )
        finished = True
    finally:
        if not finished:
            _remove_partial_experiment(experiment_dir)
    dropped = len(rows) - len(kept)
    if dropped:
        print(f"Dropped {dropped} rows with no {chosen_target}.")
    split = (
        f"{chosen_time}. Override it with --time-column <column>"
        if chosen_time
        else "file row order. Pass --time-column <column> to split by time"
    )
    print(f"Target: {chosen_target}. Split: {split}.")
    print(
        f"Lines marked CHANGE in {(experiment_dir / MODEL_DIRNAME).relative_to(root)} are the choices to review."
    )
    return experiment_dir


def _candidates_file(root: Path, experiment: str | None) -> Path:
    if experiment is not None:
        return Path(experiment) / MODEL_DIRNAME / "candidates.py"
    found = sorted(root.glob(f"experiments/*/{MODEL_DIRNAME}/candidates.py"))
    if len(found) != 1:
        raise ValueError(
            f"{root / 'experiments'}: found {len(found)} candidates.py files. "
            "Fix: pass --experiment experiments/<slug>"
        )
    return found[0]


def new_candidate(root: Path, name: str, experiment: str | None) -> tuple[Path, str]:
    """Append a working candidate, and return its file and its candidate id."""
    function = name.replace("-", "_")
    if not _NAME.match(function) or keyword.iskeyword(function):
        raise ValueError(
            f"{name} is not a usable name. Fix: use lowercase letters, digits, and `_`, "
            "and avoid Python keywords"
        )
    path = _candidates_file(root, experiment)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: no such file. Fix: run `experiment new <slug> --csv <file>` first"
        )
    text = path.read_text(encoding="utf-8")
    if f"def {function}(" in text:
        raise ValueError(f"{path}: {function} already exists. Fix: choose another name")
    write_text(
        path, text.rstrip("\n") + "\n" + _template("scaffold_candidate.py.txt", NAME=function)
    )
    return path, function.replace("_", "-")
