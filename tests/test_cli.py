from __future__ import annotations

from pathlib import Path

import pytest

from agentic_ml_loop.cli.main import main
from agentic_ml_loop.cli.runner import RunnerConfig, run_runner_main
from agentic_ml_loop.paths import DATA_DIRNAME
from agentic_ml_loop.runner import ExperimentSeed
from agentic_ml_loop.sources import SnapshotFreezeRequest, freeze_snapshot
from tests.conftest import fitted, rows


def _config(code_root: Path) -> RunnerConfig:
    return RunnerConfig(
        seed=ExperimentSeed("experiment"),
        candidate_runners={"candidate-a": fitted},
        rows_loader=lambda seed: rows(seed=seed),
        time_column="when",
        target_column="label",
        code_root=code_root,
    )


@pytest.mark.parametrize(
    "command",
    [
        ["run-candidate", "--candidate", "candidate-a"],
        ["freeze-candidate", "--candidate", "candidate-a"],
        ["evaluate-final"],
    ],
)
def test_missing_experiment_reports_one_line_and_exits_1(
    tmp_path: Path, code_root: Path, capsys, command: list[str]
) -> None:
    code = run_runner_main(_config(code_root), [*command, "--experiment", str(tmp_path / "absent")])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: Experiment directory does not exist")
    assert "Traceback" not in captured.err


def test_init_demo_seals_once(tmp_path: Path, code_root: Path, capsys) -> None:
    experiment = str(tmp_path / "run")
    assert run_runner_main(_config(code_root), ["init-demo", "--experiment", experiment]) == 0
    assert "Sealed" in capsys.readouterr().out
    assert run_runner_main(_config(code_root), ["init-demo", "--experiment", experiment]) == 0
    assert "already sealed" in capsys.readouterr().out


def test_invalid_results_report_a_concise_error(sealed: Path, code_root: Path, capsys) -> None:
    (sealed / "results.json").write_text('[{"candidate_id": "x"}]')
    code = run_runner_main(
        _config(code_root),
        ["run-candidate", "--experiment", str(sealed), "--candidate", "candidate-a"],
    )
    err = capsys.readouterr().err
    assert code == 1
    assert err.startswith("error: invalid")
    assert "objective_score" in err


def test_validate_failure_goes_to_stderr(tmp_path: Path, capsys) -> None:
    assert main(["validate", str(tmp_path / "absent")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "does not exist" in captured.err


def test_replay_on_a_missing_experiment_creates_nothing(tmp_path: Path, capsys) -> None:
    absent = tmp_path / "absent"
    assert main(["replay", str(absent)]) == 1
    assert "does not exist" in capsys.readouterr().err
    assert not absent.exists()


def _pulled(tmp_path: Path) -> Path:
    pyarrow = pytest.importorskip("pyarrow")

    class StaticExtractor:
        source_type = "duckdb"
        driver = "static"

        def extract(self, query: str):
            return pyarrow.Table.from_pandas(rows(), preserve_index=False)

    experiment = tmp_path / "pulled"
    freeze_snapshot(
        SnapshotFreezeRequest(
            source_type="duckdb",
            query="select * from t order by id",
            out_dir=experiment / DATA_DIRNAME,
            extractor=StaticExtractor(),
        )
    )
    return experiment


def test_seal_moves_snapshot_rows_out_of_the_workspace(tmp_path: Path, capsys) -> None:
    experiment = _pulled(tmp_path)
    assert main(["seal", "--experiment", str(experiment), "--time-column", "when"]) == 0
    assert "Sealed 12 holdout rows" in capsys.readouterr().out
    assert not (experiment / DATA_DIRNAME / "snapshot.parquet").exists()
    assert (experiment / DATA_DIRNAME / "train.parquet").is_file()


def test_pull_after_seal_is_refused_and_crash_recovery_finishes(tmp_path: Path, capsys) -> None:
    experiment = _pulled(tmp_path)
    snapshot = (experiment / DATA_DIRNAME / "snapshot.parquet").read_bytes()
    assert main(["seal", "--experiment", str(experiment), "--time-column", "when"]) == 0
    capsys.readouterr()

    assert (
        main(["pull", "--experiment", str(experiment), "--source", "duckdb", "--query", "select 1"])
        == 1
    )
    assert "This experiment is sealed" in capsys.readouterr().err
    assert not (experiment / DATA_DIRNAME / "snapshot.parquet").exists()

    # A crash before the removal leaves the snapshot beside the seal.
    (experiment / DATA_DIRNAME / "snapshot.parquet").write_bytes(snapshot)
    (experiment / "experiment.md").write_text("# x\n")
    (experiment / "research_journal.md").write_text("# x\n")
    (experiment / "results.json").write_text("[]\n")
    assert main(["validate", str(experiment)]) == 1
    assert "beside a seal" in capsys.readouterr().err
    assert main(["seal", "--experiment", str(experiment), "--time-column", "when"]) == 0
    assert not (experiment / DATA_DIRNAME / "snapshot.parquet").exists()
    assert main(["validate", str(experiment)]) == 0


def _duckdb_file(tmp_path: Path) -> Path:
    duckdb = pytest.importorskip("duckdb")
    database = tmp_path / "x.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute("create table t as select 1 as id")
    connection.close()
    return database


def test_pull_creates_a_missing_directory(tmp_path: Path) -> None:
    experiment = tmp_path / "new" / "run"
    code = main(
        [
            "pull",
            "--experiment",
            str(experiment),
            "--source",
            "duckdb",
            "--database",
            str(_duckdb_file(tmp_path)),
            "--query",
            "select id from t order by id",
        ]
    )
    assert code == 0
    assert (experiment / DATA_DIRNAME / "snapshot.parquet").is_file()


def test_driver_error_is_one_line_without_a_traceback(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "pull",
            "--experiment",
            str(tmp_path / "run"),
            "--source",
            "duckdb",
            "--database",
            str(_duckdb_file(tmp_path)),
            "--query",
            "select id from missing_table order by id",
        ]
    )
    err = capsys.readouterr().err
    assert code == 1
    assert err.startswith("error: duckdb query failed")
    assert err.count("\n") == 1


def test_sources_list_runs_offline(capsys) -> None:
    assert main(["sources", "list"]) == 0
    assert "duckdb" in capsys.readouterr().out
