from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from agentic_ml_loop.cli.main import CONTRACT_DOCUMENTS, main
from agentic_ml_loop.cli.runner import RunnerConfig, run_runner_main
from agentic_ml_loop.cli.setup import PACKAGED_SKILLS
from agentic_ml_loop.holdout import load_search_splits
from agentic_ml_loop.ledger import store_root
from agentic_ml_loop.paths import DATA_DIRNAME
from agentic_ml_loop.runner import ExperimentSeed
from agentic_ml_loop.sources import SnapshotFreezeRequest, freeze_snapshot
from tests.conftest import fitted, rows


def _config(code_root: Path) -> RunnerConfig:
    return RunnerConfig(
        seed=ExperimentSeed("experiment"),
        candidate_runners={"candidate-a": fitted},
        rows_loader=lambda seed: rows(seed=seed),
        rows_source=None,
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


def _skill_files() -> list[Path]:
    return [path for path in PACKAGED_SKILLS.rglob("*.md") if path.is_file()]


def test_setup_links_packaged_skills_and_keeps_local_ones(tmp_path: Path, capsys) -> None:
    local = tmp_path / ".agents" / "skills" / "research"
    local.mkdir(parents=True)
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    assert "kept local skill research" in capsys.readouterr().out
    assert not local.is_symlink()
    link = tmp_path / ".claude" / "skills" / "experiment-spec" / "SKILL.md"
    assert link.read_text().startswith("---\nname: experiment-spec")
    assert not (tmp_path / ".agents/skills/experiment-spec").readlink().is_absolute()


def test_second_setup_changes_nothing(tmp_path: Path, capsys) -> None:
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    assert capsys.readouterr().out == "nothing to do\n"


def test_setup_replaces_a_link_to_an_older_install(tmp_path: Path) -> None:
    skills = tmp_path / ".agents" / "skills"
    skills.mkdir(parents=True)
    (skills / "research").symlink_to("../../old/agentic_ml_loop/skills/research")
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    assert (skills / "research" / "SKILL.md").is_file()


@pytest.mark.parametrize("document", sorted(CONTRACT_DOCUMENTS))
def test_contract_prints_each_packaged_document(document: str, capsys) -> None:
    assert main(["contract", document]) == 0
    assert capsys.readouterr().out.startswith("#")


def test_packaged_skills_need_no_checkout() -> None:
    needs_checkout = re.compile(r"\.\./\.\./|src/agentic_ml_loop|\]\(docs/|\]\(demos/")
    assert [path.name for path in _skill_files() if needs_checkout.search(path.read_text())] == []


def test_seal_names_the_local_file_that_keeps_the_rows(
    tmp_path: Path, code_root: Path, capsys
) -> None:
    source = tmp_path / "rows.csv"
    config = replace(_config(code_root), rows_source=source)
    assert run_runner_main(config, ["init-demo", "--experiment", str(tmp_path / "e")]) == 0
    assert f"stay readable in {source}" in capsys.readouterr().err


def test_seal_with_no_local_file_does_not_warn(tmp_path: Path, code_root: Path, capsys) -> None:
    assert (
        run_runner_main(_config(code_root), ["init-demo", "--experiment", str(tmp_path / "e")]) == 0
    )
    assert "stay readable" not in capsys.readouterr().err


def test_reseal_names_the_rows_file(scored: Path, tmp_path: Path, capsys) -> None:
    source = tmp_path / "new_rows.parquet"
    rows(count=12, seed=99, start="2026-06-01").to_parquet(source, index=False)
    arguments = ["reseal", "--experiment", str(scored), "--rows", str(source), "--note", "New."]
    assert main(arguments) == 0
    assert "stay readable in" in capsys.readouterr().err


def _provenance(artifact: Path, label_date: str) -> None:
    artifact.write_text("feature\nsignal\n")
    (artifact.parent / "feature_selection_provenance.json").write_text(
        json.dumps(
            {
                artifact.name: {
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "label_dates": [label_date],
                }
            }
        )
    )


def test_check_features_reads_dates_from_search_rows(sealed: Path, capsys) -> None:
    artifact = sealed / "work" / "ranked.csv"
    splits = load_search_splits(sealed)
    arguments = [
        "check-features",
        "--experiment",
        str(sealed),
        "--artifact",
        str(artifact),
        "--time-column",
        "when",
    ]
    _provenance(artifact, splits.train["when"].iloc[0].strftime("%Y-%m-%d"))
    assert main(arguments) == 0
    _provenance(artifact, splits.validation["when"].iloc[0].strftime("%Y-%m-%d"))
    assert main(arguments) == 1
    assert "validation or holdout labels" in capsys.readouterr().err
    _provenance(artifact, "2031-01-01")
    assert main(arguments) == 1
    assert "outside selection training dates" in capsys.readouterr().err
    assert main([*arguments[:-1], "absent"]) == 1
    assert "Time column is missing" in capsys.readouterr().err


def test_setup_merges_the_deny_rule_and_keeps_other_settings(tmp_path: Path, capsys) -> None:
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir()
    settings_path.write_text(
        json.dumps({"model": "x", "permissions": {"allow": ["Bash(ls)"], "deny": ["Read(./.env)"]}})
    )
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    settings = json.loads(settings_path.read_text())
    store = store_root().resolve().as_posix()
    assert settings["model"] == "x"
    assert settings["permissions"]["allow"] == ["Bash(ls)"]
    assert settings["permissions"]["deny"] == [
        "Read(./.env)",
        f"Read(/{store}/**)",
        f"Edit(/{store}/**)",
        "Bash(*agentic-ml-loop/holdouts*)",
    ]
    assert store.startswith("/") and settings["permissions"]["deny"][1].startswith("Read(//")
    first = settings_path.read_text()
    capsys.readouterr()
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    assert settings_path.read_text() == first
    assert capsys.readouterr().out == "nothing to do\n"


@pytest.mark.parametrize(
    "content", ["not json", "[]", '{"permissions": []}', '{"permissions": {"deny": "x"}}']
)
def test_setup_refuses_settings_it_cannot_merge(tmp_path: Path, content: str, capsys) -> None:
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir()
    settings_path.write_text(content)
    assert main(["setup", "--repo", str(tmp_path)]) == 1
    assert settings_path.read_text() == content
    assert "Cannot add holdout deny rules" in capsys.readouterr().err
    assert not (tmp_path / ".agents").exists()


def _deny_rules(repo: Path) -> list[str]:
    return json.loads((repo / ".claude/settings.json").read_text())["permissions"]["deny"]


def test_store_under_home_gets_a_portable_rule(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("XDG_DATA_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert main(["setup", "--repo", str(tmp_path / "repo")]) == 0
    rule = "~/.local/share/agentic-ml-loop/holdouts/**"
    assert _deny_rules(tmp_path / "repo")[:2] == [f"Read({rule})", f"Edit({rule})"]


def test_another_store_replaces_the_engine_rules_and_keeps_user_rules(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    user_rules = ["Read(./.env)", "Read(//srv/secrets/**)", "Read(~/agentic-ml-loop/notes/**)"]
    settings = repo / ".claude/settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"permissions": {"deny": user_rules}}))
    assert main(["setup", "--repo", str(repo)]) == 0
    first_store = store_root().resolve().as_posix()

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "other"))
    assert main(["setup", "--repo", str(repo)]) == 0
    second_store = store_root().resolve().as_posix()

    rules = _deny_rules(repo)
    assert first_store != second_store
    assert not [rule for rule in rules if first_store in rule]
    assert rules == [
        *user_rules,
        "Bash(*agentic-ml-loop/holdouts*)",
        f"Read(/{second_store}/**)",
        f"Edit(/{second_store}/**)",
    ]
