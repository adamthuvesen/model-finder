"""`experiment next`: read the state, and suggest useful steps. It is advice, and it blocks nothing."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from model_finder.completion import assess_completion
from model_finder.holdout import is_sealed
from model_finder.paths import (
    ATTEMPT_PATH,
    COMPLETION_PATH,
    FINAL_PATH,
    MODEL_DIRNAME,
    RUNNER_COMMAND_PATH,
    SELECTION_PATH,
)
from model_finder.results import load_results, ranked_results


@dataclass(frozen=True)
class Step:
    text: str
    command: str


def command_prefix() -> str:
    """Make a printed command run when pasted: in a uv project, or from a local environment."""
    if Path("uv.lock").is_file():
        return "uv run "
    return ".venv/bin/" if Path(".venv/bin/experiment").exists() else ""


def _runner_command(experiment_dir: Path, prefix: str) -> str:
    recorded = experiment_dir / RUNNER_COMMAND_PATH
    if recorded.is_file():
        return prefix + recorded.read_text(encoding="utf-8").strip()
    return "<your runner>"


def _unused_candidate_name(experiment_dir: Path) -> str:
    path = experiment_dir / MODEL_DIRNAME / "candidates.py"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    number = 2
    while f"def candidate_{number}(" in text:
        number += 1
    return f"candidate_{number}"


def choose_experiment(root: Path) -> Path | None:
    found = sorted(path.parent for path in root.glob("experiments/*/experiment.md"))
    return found[0] if len(found) == 1 else None


def _search_steps(
    experiment_dir: Path, runner: str, target: str, quoted: str, tool: str
) -> list[Step]:
    """Suggest what fits once there are results. Search stays open in any order."""
    best = ranked_results(load_results(experiment_dir))[0].candidate_id
    has_seal = is_sealed(experiment_dir)
    if (experiment_dir / MODEL_DIRNAME / "candidates.py").is_file():
        name = _unused_candidate_name(experiment_dir)
        another = Step(
            f"Try another candidate. Edit it, then run `{runner} run-candidate {target}--candidate "
            f"{name.replace('_', '-')}`.",
            f"{tool}experiment candidate new {name} --experiment {quoted}",
        )
    else:
        another = Step(
            "Run a candidate again after you change it.",
            f"{runner} run-candidate {target}--candidate {best}",
        )
    record = Step(
        "Record what you concluded. Replace the conclusion text.",
        f"{tool}experiment record-completion --experiment {quoted} --outcome supported "
        '--conclusion "CHANGE: what the evidence supports"',
    )
    if (experiment_dir / COMPLETION_PATH).is_file():
        label = assess_completion(experiment_dir).label
        check = Step(
            f"This work reads: {label}. See why.", f"{tool}experiment complete-check {quoted}"
        )
        return [check, another]
    if (experiment_dir / FINAL_PATH).is_file():
        return [record]
    if (experiment_dir / ATTEMPT_PATH).is_file():
        retry = Step(
            "The last evaluation did not finish. Retry it, which labels it exploratory.",
            f'{runner} evaluate-final {target}--allow-reread --reason "CHANGE: why"',
        )
        return [retry, record]
    if has_seal and (experiment_dir / SELECTION_PATH).is_file():
        final = Step(
            "Score the frozen candidate on the holdout.",
            f"{runner} evaluate-final {target}".rstrip(),
        )
        return [final, another]
    if has_seal:
        freeze = Step(
            f"Freeze the best so far, {best}, when you are done searching.",
            f"{runner} freeze-candidate {target}--candidate {best}",
        )
        return [another, freeze, record]
    holdout = Step(
        "Add a holdout for an unbiased estimate. It needs a parquet file of unseen rows.",
        f"{tool}experiment reseal --experiment {quoted} --rows new_rows.parquet "
        '--note "CHANGE: where the rows came from"',
    )
    return [another, record, holdout]


def next_steps(experiment_dir: Path) -> list[Step]:
    """Return the suggested step first, and the alternatives after it."""
    tool = command_prefix()
    if not experiment_dir.is_dir():
        return [Step("Start an experiment.", f"{tool}experiment new my_run --csv rows.csv")]
    shown = (
        experiment_dir.relative_to(Path.cwd())
        if experiment_dir.is_absolute() and experiment_dir.is_relative_to(Path.cwd())
        else experiment_dir
    )
    quoted = shlex.quote(str(shown))
    runner = _runner_command(experiment_dir, tool)
    has_default = (experiment_dir / MODEL_DIRNAME / "run.py").is_file()
    target = "" if has_default else f"--experiment {quoted} "
    if not load_results(experiment_dir):
        first = Step(
            "Run a first candidate.", f"{runner} run-candidate {target}--candidate baseline"
        )
        return [first]
    return _search_steps(experiment_dir, runner, target, quoted, tool)
