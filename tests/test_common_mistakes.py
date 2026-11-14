"""Each test reproduces a reviewer's finding: a misleading number, a failed path, or a bad message."""

from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_finder.candidates import Candidates, Scoring, scored_candidate
from model_finder.cli.errors import run_reporting_errors
from model_finder.cli.hook import edited_files, report
from model_finder.cli.main import main
from model_finder.cli.next_step import next_steps
from model_finder.cli.runner import run_runner_main
from model_finder.cli.scaffold import load_runner
from model_finder.metrics import classification_scores, regression_scores
from model_finder.sources import adapters, registry
from model_finder.splits import SearchSplits, temporal_split

ROOT = Path(__file__).parents[1]


def _frame(count: int = 200, *, date: str = "%Y-%m-%d", seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=count)
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=count, freq="7D").strftime(date),
            "signal": signal,
            "label": (signal + rng.normal(size=count) > 0).astype(float),
        }
    )


@pytest.fixture
def consumer(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    _frame().to_csv("rows.csv", index=False)
    return tmp_path


def _splits() -> SearchSplits:
    return SearchSplits(train=_frame(seed=1), validation=_frame(seed=2))


@pytest.mark.parametrize("scores", [classification_scores, regression_scores])
def test_metrics_refuse_missing_labels(scores) -> None:
    target = pd.Series([0.0, 1.0, np.nan], name="label")
    with pytest.raises(ValueError, match=r"label has 1 missing.*Fix:"):
        scores(target, np.array([0.1, 0.9, 0.5]))


def test_auc_refuses_labels_that_are_not_binary() -> None:
    with pytest.raises(ValueError, match=r"label.*not 0 and 1.*Fix:"):
        classification_scores(pd.Series([0.0, 1.0, 2.0], name="label"), np.array([0.1, 0.5, 0.9]))


def test_new_drops_rows_with_a_missing_target_and_says_how_many(consumer: Path, capsys) -> None:
    frame = _frame()
    frame.loc[frame.index[:7], "label"] = np.nan
    frame.to_csv("gaps.csv", index=False)
    assert main(["new", "gaps", "--csv", "gaps.csv"]) == 0
    assert "Dropped 7 rows with no label" in capsys.readouterr().out
    train = pd.read_parquet("experiments/gaps/data/train.parquet")
    assert train["label"].notna().all()


def test_time_sorts_on_parsed_dates_across_years() -> None:
    frame = _frame(date="%m/%d/%Y").sample(frac=1.0, random_state=3)
    search, holdout = temporal_split(frame, "date")
    train_last = pd.to_datetime(search.train["date"], format="%m/%d/%Y").max()
    validation_first = pd.to_datetime(search.validation["date"], format="%m/%d/%Y").min()
    assert train_last < validation_first
    assert pd.to_datetime(holdout.holdout["date"], format="%m/%d/%Y").min() > train_last


def test_a_time_column_that_does_not_parse_names_the_column_and_fix() -> None:
    frame = _frame().assign(date=[f"week {index}" for index in range(200)])
    with pytest.raises(ValueError, match=r"date.*does not parse.*Fix:"):
        temporal_split(frame, "date")


def test_numeric_time_named_column_stays_a_feature(consumer: Path, capsys) -> None:
    frame = _frame().drop(columns="date").assign(response_time=lambda rows: rows["signal"])
    frame.to_csv("numeric.csv", index=False)
    assert main(["new", "numeric", "--csv", "numeric.csv", "--target", "label"]) == 0
    out = capsys.readouterr().out
    assert "Split: file row order" in out
    assert "--time-column" in out
    assert "response_time" not in Path("experiments/numeric/model/candidates.py").read_text()


def test_auto_selected_date_column_is_named_with_its_override(consumer: Path, capsys) -> None:
    assert main(["new", "dated", "--csv", "rows.csv"]) == 0
    out = capsys.readouterr().out
    assert "Split: date" in out
    assert "--time-column" in out


def test_predict_never_receives_the_target() -> None:
    seen: list[list[str]] = []

    def predict(frame: pd.DataFrame) -> np.ndarray:
        seen.append(list(frame.columns))
        return frame.select_dtypes("number").sum(axis=1).to_numpy()

    fitted = scored_candidate(
        _splits(), Scoring.classification("label"), predict, candidate_id="leaky"
    )
    assert fitted.score_rows(_frame(seed=3))["auc"] < 0.95
    assert all("label" not in columns for columns in seen)


def test_a_failed_new_leaves_nothing_behind(consumer: Path, capsys) -> None:
    _frame(count=3).to_csv("tiny.csv", index=False)
    assert main(["new", "tiny", "--csv", "tiny.csv"]) == 1
    assert "too small" in capsys.readouterr().err
    assert not Path("experiments/tiny").exists()
    _frame().to_csv("tiny.csv", index=False)
    assert main(["new", "tiny", "--csv", "tiny.csv"]) == 0


def test_next_never_prints_the_console_script_as_the_runner(consumer: Path) -> None:
    assert main(["new", "churn", "--csv", "rows.csv"]) == 0
    text = " ".join(step.command for step in next_steps(Path("experiments/churn")))
    assert "bin/experiment" not in text
    assert "experiments/churn/model/run.py" in text


def _literal_commands(experiment: Path) -> list[list[str]]:
    return [shlex.split(step.command) for step in next_steps(experiment)]


def test_every_suggestion_runs_when_pasted(consumer: Path) -> None:
    assert main(["new", "churn", "--csv", "rows.csv", "--no-holdout"]) == 0
    experiment = Path("experiments/churn")
    config = load_runner(experiment / "model")
    assert run_runner_main(config, ["freeze-candidate", "--candidate", "linear"]) == 0
    text = " ".join(step.command for step in next_steps(experiment))
    assert "evaluate-final" not in text
    for command in _literal_commands(experiment):
        assert "<" not in " ".join(command), command


def test_suggestions_use_uv_run_in_a_uv_project(consumer: Path) -> None:
    assert main(["new", "churn", "--csv", "rows.csv"]) == 0
    Path("uv.lock").write_text("")
    for step in next_steps(Path("experiments/churn")):
        assert step.command.startswith("uv run "), step.command


@pytest.mark.parametrize("command", ["leaderboard", "next"])
def test_a_missing_experiment_fails_and_says_what_was_looked_for(
    command: str, consumer: Path, capsys
) -> None:
    assert main([command, "nope"]) == 1
    err = capsys.readouterr().err
    assert "nope" in err
    assert "Fix:" in err


def _candidate_file(tmp_path: Path, body: str) -> Candidates:
    source = tmp_path / "user_candidates.py"
    source.write_text(body)
    namespace: dict[str, object] = {"Candidates": Candidates, "Scoring": Scoring, "np": np}
    exec(compile(source.read_text(), str(source), "exec"), namespace)  # noqa: S102
    return namespace["candidates"]  # pyright: ignore[reportReturnType] -- the exec namespace holds object values


HEADER = 'candidates = Candidates(Scoring.classification("label"))\n\n'


def _failure(capsys, candidates: Candidates) -> str:
    def command() -> int:
        candidates[next(iter(candidates))](_splits())
        return 0

    assert run_reporting_errors(command) == 1
    return capsys.readouterr().err


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            "@candidates.add\ndef model(splits):\n    return object()\n",
            r"user_candidates\.py:\d+: .*did not return a function.*Fix: return",
        ),
        (
            "@candidates.add\ndef flat(splits):\n"
            "    return lambda rows: np.zeros((len(rows), 2))\n",
            r"user_candidates\.py:\d+: .*one value per row.*Fix:.*\[:, 1\]",
        ),
        (
            "@candidates.add\ndef short(splits):\n    return lambda rows: np.zeros(3)\n",
            r"user_candidates\.py:\d+: .*3 predictions for 200 rows.*Fix:",
        ),
        (
            '@candidates.add\ndef column(splits):\n    return splits.train["nope"]\n',
            r"user_candidates\.py:\d+: .*nope.*Fix: use one of: date, signal, label",
        ),
        (
            "@candidates.add\ndef broken(splits):\n    return 1 / 0\n",
            r"user_candidates\.py:5: ZeroDivisionError: division by zero",
        ),
        (
            "@candidates.add\ndef typo(splits):\n    return splits.nope\n",
            r"user_candidates\.py:5: AttributeError: .*nope",
        ),
        (
            "@candidates.add\ndef index(splits):\n    return [][1]\n",
            r"user_candidates\.py:5: IndexError: ",
        ),
    ],
)
def test_a_mistake_in_a_candidate_reports_its_own_line(
    body: str, expected: str, tmp_path: Path, capsys
) -> None:
    err = _failure(capsys, _candidate_file(tmp_path, HEADER + body))
    assert __import__("re").search(expected, err), err
    assert "Traceback" not in err
    assert "experiment next" not in err


def test_a_bad_metric_lists_the_valid_ones(capsys) -> None:
    scoring = Scoring("label", "val_nope", lambda _target, _scores: {"auc": 0.5, "f1": 0.5})

    def command() -> int:
        scored_candidate(_splits(), scoring, lambda rows: np.zeros(len(rows)), candidate_id="c")
        return 0

    assert run_reporting_errors(command) == 1
    err = capsys.readouterr().err
    assert "val_nope" in err
    assert "val_auc" in err


def test_a_missing_extra_keeps_only_its_own_fix() -> None:
    from model_finder.cli.errors import describe
    from model_finder.sources.errors import MissingSourceExtra

    message = describe(MissingSourceExtra("postgres", "postgres", "adbc"))
    assert "uv sync --extra postgres" in message
    assert "experiment next" not in message


def test_a_nan_metric_names_the_real_fix() -> None:
    from pydantic import ValidationError

    from model_finder.candidate_result import SearchMetrics
    from model_finder.cli.errors import describe

    with pytest.raises(ValidationError) as caught:
        SearchMetrics(validation={"auc": float("nan")})
    message = describe(caught.value)
    assert "return a real number" not in message
    assert "missing values" in message


def test_an_engine_bug_keeps_its_traceback() -> None:
    def command() -> int:
        raise AttributeError("engine bug")

    command.__code__ = command.__code__.replace(
        co_filename=str(ROOT / "src/model_finder/cli/main.py")
    )
    with pytest.raises(AttributeError):
        run_reporting_errors(command)


def test_a_duplicate_candidate_names_both_locations(tmp_path: Path) -> None:
    body = HEADER + (
        "@candidates.add\ndef same(splits):\n    return None\n\n"
        "@candidates.add\ndef same(splits):\n    return None\n"
    )
    with pytest.raises(ValueError, match=r"user_candidates\.py:7.*user_candidates\.py:3.*Fix:"):
        _candidate_file(tmp_path, body)


@pytest.mark.parametrize("name", ["class", "2fast", "a b", "for"])
def test_candidate_new_rejects_unusable_names(name: str, consumer: Path, capsys) -> None:
    assert main(["new", "churn", "--csv", "rows.csv"]) == 0
    path = Path("experiments/churn/model/candidates.py")
    before = path.read_text()
    assert main(["candidate", "new", name]) == 1
    assert "Fix:" in capsys.readouterr().err
    assert path.read_text() == before


@pytest.mark.parametrize("slug", ["../outside", "a/b", 'a"b', ".hidden", ""])
def test_new_rejects_a_slug_that_is_not_a_plain_name(slug: str, consumer: Path, capsys) -> None:
    assert main(["new", slug, "--csv", "rows.csv"]) == 1
    assert "Fix:" in capsys.readouterr().err
    assert not Path("outside").exists()
    assert not Path("experiments").exists()


def test_a_target_with_a_quote_still_generates_valid_code(consumer: Path) -> None:
    _frame().rename(columns={"label": 'x"y'}).to_csv("odd.csv", index=False)
    assert main(["new", "odd", "--csv", "odd.csv", "--target", 'x"y']) == 0
    assert load_runner(Path("experiments/odd/model")).target_column == 'x"y'


def test_generated_files_pass_the_format_check(consumer: Path) -> None:
    assert main(["new", "churn", "--csv", "rows.csv"]) == 0
    assert main(["candidate", "new", "wide_model"]) == 0
    done = subprocess.run(
        [
            str(Path(sys.executable).parent / "ruff"),
            "format",
            "--check",
            "--isolated",
            "experiments/churn/model",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout


def _bad_bundle(tmp_path: Path, monkeypatch, name: str, content: str) -> None:
    bundles = tmp_path / "bundles"
    (bundles / name).mkdir(parents=True)
    (bundles / name / "source.json").write_text(content)
    monkeypatch.setattr(registry, "BUNDLES_DIR", bundles)


def test_a_malformed_bundle_is_reported_and_does_not_break_the_rest(
    tmp_path: Path, monkeypatch
) -> None:
    _bad_bundle(tmp_path, monkeypatch, "broken", "{ not json")
    assert registry.bundle_definitions() == []
    problems = registry.bundle_problems()
    assert any("broken/source.json" in problem and "Fix:" in problem for problem in problems)


def test_a_bundle_with_no_dialect_extra_or_adapter_is_reported(tmp_path: Path, monkeypatch) -> None:
    fields = {
        "source_type": "mysqlx",
        "display_name": "X",
        "extra": "mysqlx",
        "package": "x",
        "driver": "adbc",
        "description": "x",
    }
    _bad_bundle(tmp_path, monkeypatch, "mysqlx", json.dumps(fields))
    problems = " ".join(registry.bundle_problems())
    assert "dialect" in problems
    assert "extra" in problems
    assert "adapter" in problems


def test_the_shipped_bundles_pass_the_check(capsys) -> None:
    assert registry.bundle_problems() == []
    assert main(["sources", "check"]) == 0
    assert adapters.SUPPORTED_SOURCE_TYPES


def test_sources_check_fails_on_a_bad_bundle(tmp_path: Path, monkeypatch, capsys) -> None:
    _bad_bundle(tmp_path, monkeypatch, "broken", "{ not json")
    assert main(["sources", "check"]) == 1
    assert "Fix:" in capsys.readouterr().err


def test_hook_output_is_capped_with_the_remaining_count_and_a_command() -> None:
    text = json.loads(report("claude", "x" * 40000))["hookSpecificOutput"]["additionalContext"]
    assert len(text) < 2300
    assert "38000 more characters" in text.splitlines()[-1]
    assert "ruff check" in text.splitlines()[-1]


def test_files_outside_the_repo_are_not_reported(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "other.py").write_text("")
    event: dict[str, object] = {"tool_input": {"file_path": str(tmp_path / "other.py")}}
    assert edited_files(event, root) == []
    inside: dict[str, object] = {"tool_input": {"file_path": str(root / "a.py")}}
    assert edited_files(inside, root) == [str((root / "a.py").resolve())]


def test_quick_handles_spaces_in_file_names(tmp_path: Path, monkeypatch) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "with space.py").write_text("VALUE = 1\n")
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("quick", ROOT / "scripts/quick.py")
    assert spec is not None
    assert spec.loader is not None
    quick = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quick)
    files = quick._changed_python_files()
    assert files == ["with space.py"]
    assert all(files == command[-len(files) :] for command in quick._tool_commands(files))


def test_a_holdout_metric_names_the_scoring_function_once(capsys) -> None:
    def evaluate(_target: pd.Series, _scores: np.ndarray) -> dict[str, float]:
        return {"auc": 0.5, "holdout_auc": 0.9}

    scoring = Scoring("label", "val_auc", evaluate)

    def command() -> int:
        scored_candidate(_splits(), scoring, lambda rows: np.zeros(len(rows)), candidate_id="c")
        return 0

    assert run_reporting_errors(command) == 1
    err = capsys.readouterr().err
    assert err.count("looks like a holdout score") == 1
    assert __import__("re").match(r"tests/test_common_mistakes\.py:\d+: metric holdout_auc", err), (
        err
    )


def test_a_repeated_validation_message_prints_once() -> None:
    from pydantic import ValidationError

    from model_finder.candidate_result import SearchMetrics
    from model_finder.cli.errors import describe

    with pytest.raises(ValidationError) as caught:
        SearchMetrics(train={"holdout_auc": 0.9}, validation={"holdout_auc": 0.9})
    assert describe(caught.value).count("looks like a holdout score") == 1


def test_a_holdout_score_in_hyperparameters_fails_strict_with_a_fix(consumer: Path, capsys) -> None:
    assert main(["new", "churn", "--csv", "rows.csv"]) == 0
    from model_finder.candidate_result import (
        CandidateResult,
        FittedCandidate,
        SearchMetrics,
    )
    from model_finder.runner import save_candidate_result

    def smuggle(_splits: SearchSplits) -> FittedCandidate:
        result = CandidateResult(
            candidate_id="smuggle",
            objective_metric="val_auc",
            objective_score=0.7,
            metrics=SearchMetrics(validation={"auc": 0.7}),
            hyperparameters={"holdout": {"rmse": 0.1}},
        )
        return FittedCandidate(result=result, score_rows=lambda _rows: {"auc": 0.7})

    experiment = Path("experiments/churn")
    save_candidate_result(experiment, "smuggle", {"smuggle": smuggle}, experiment / "model")
    arguments = ["record-completion", "--experiment", str(experiment), "--outcome", "inconclusive"]
    assert main([*arguments, "--conclusion", "Unclear."]) == 0
    capsys.readouterr()
    assert main(["complete-check", str(experiment), "--strict"]) == 1
    captured = capsys.readouterr()
    assert "holdout" in captured.out
    line = next(text for text in captured.out.splitlines() if "hyperparameter" in text)
    assert "Fix:" in line


def test_uv_keeps_the_environment_under_a_custom_data_home() -> None:
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["tool"]["uv"]["python-preference"] == "managed"
