from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from model_finder.cli.main import CONTRACT_DOCUMENTS, main
from model_finder.cli.runner import RunnerConfig, run_runner_main
from model_finder.cli.setup import PACKAGED_SKILLS
from model_finder.final_evaluation import freeze_candidate
from model_finder.holdout import load_search_splits
from model_finder.paths import DATA_DIRNAME
from model_finder.runner import ExperimentSeed, save_candidate_result
from model_finder.sources import SnapshotFreezeRequest, freeze_snapshot
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
    assert "experiment directory does not exist. Fix:" in captured.err
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
    assert err.startswith("results.json: ")
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
    assert err.startswith("duckdb query failed")
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
    (skills / "research").symlink_to("../../old/model_finder/skills/research")
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    assert (skills / "research" / "SKILL.md").is_file()


@pytest.mark.parametrize("document", sorted(CONTRACT_DOCUMENTS))
def test_contract_prints_each_packaged_document(document: str, capsys) -> None:
    assert main(["contract", document]) == 0
    assert capsys.readouterr().out.startswith("#")


def test_packaged_skills_need_no_checkout() -> None:
    needs_checkout = re.compile(r"\.\./\.\./|src/model_finder|\]\(docs/|\]\(demos/")
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


DENY_RULES = [
    "Read(//**/model-finder/holdouts/**)",
    "Edit(//**/model-finder/holdouts/**)",
    "Bash(*model-finder/holdouts*)",
    "Read(//**/agentic-ml-loop/holdouts/**)",
    "Edit(//**/agentic-ml-loop/holdouts/**)",
    "Bash(*agentic-ml-loop/holdouts*)",
]


def test_setup_merges_the_deny_rule_and_keeps_other_settings(tmp_path: Path, capsys) -> None:
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir()
    settings_path.write_text(
        json.dumps({"model": "x", "permissions": {"allow": ["Bash(ls)"], "deny": ["Read(./.env)"]}})
    )
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    settings = json.loads(settings_path.read_text())
    assert settings["model"] == "x"
    assert settings["permissions"]["allow"] == ["Bash(ls)"]
    assert settings["permissions"]["deny"] == ["Read(./.env)", *DENY_RULES]
    first = settings_path.read_text()
    capsys.readouterr()
    assert main(["setup", "--repo", str(tmp_path)]) == 0
    assert settings_path.read_text() == first
    assert capsys.readouterr().out == "nothing to do\n"


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        '{"permissions": []}',
        '{"permissions": {"deny": "x"}}',
        '{"permissions": {"deny": [3]}}',
        '{"model": "x", "permissions": {"deny": ["Read(./.env)", null]}}',
    ],
)
def test_setup_refuses_settings_it_cannot_merge(tmp_path: Path, content: str, capsys) -> None:
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir()
    settings_path.write_text(content)
    assert main(["setup", "--repo", str(tmp_path)]) == 1
    assert settings_path.read_text() == content
    captured = capsys.readouterr()
    assert "Cannot add holdout deny rules" in captured.err
    assert str(settings_path) in captured.err
    assert "Fix:" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    if '"deny": [3]' in content:
        assert "permissions.deny[0] must be a string" in captured.err
    if "null" in content:
        assert "permissions.deny[1] must be a string" in captured.err
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == [
        ".claude",
        ".claude/settings.json",
    ]
    assert not (tmp_path / ".agents").exists()


def _deny_rules(repo: Path) -> list[str]:
    return json.loads((repo / ".claude/settings.json").read_text())["permissions"]["deny"]


def test_setup_writes_the_same_bytes_on_any_machine(tmp_path: Path, monkeypatch) -> None:
    written: list[str] = []
    for name, environment in (
        ("home", {"HOME": str(tmp_path / "home_a")}),
        ("other", {"HOME": str(tmp_path / "home_b"), "XDG_DATA_HOME": str(tmp_path / "custom")}),
    ):
        monkeypatch.delenv("XDG_DATA_HOME")
        for key, value in environment.items():
            monkeypatch.setenv(key, value)
        repo = tmp_path / name
        assert main(["setup", "--repo", str(repo)]) == 0
        written.append((repo / ".claude/settings.json").read_text())
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "unused"))
    assert written[0] == written[1]
    assert _deny_rules(tmp_path / "home") == DENY_RULES
    assert str(tmp_path) not in written[0]


def test_setup_replaces_rules_from_an_earlier_version_and_keeps_user_rules(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    user_rules = ["Read(./.env)", "Read(//srv/secrets/**)", "Read(~/model-finder/notes/**)"]
    earlier = [
        "Read(~/.local/share/model-finder/holdouts/**)",
        "Edit(~/.local/share/model-finder/holdouts/**)",
        "Read(//Users/someone/custom/model-finder/holdouts/**)",
        "Edit(//Users/someone/custom/model-finder/holdouts/**)",
    ]
    settings = repo / ".claude/settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"permissions": {"deny": [*user_rules, *earlier]}}))
    assert main(["setup", "--repo", str(repo)]) == 0
    assert _deny_rules(repo) == [*user_rules, *DENY_RULES]


_TWO_RUNNERS = {
    "candidate-a": fitted,
    "candidate-b": lambda splits: fitted(splits, "candidate-b"),
}


def _two_candidates(sealed: Path, code_root: Path) -> Path:
    for candidate in ("candidate-a", "candidate-b", "candidate-a"):
        save_candidate_result(sealed, candidate, _TWO_RUNNERS, code_root)
    return sealed


def _rebuild(path: Path) -> int:
    return main(["rebuild-results", str(path)])


def test_damaged_leaderboard_is_rebuilt_from_run_records(
    sealed: Path, code_root: Path, capsys
) -> None:
    _two_candidates(sealed, code_root)
    leaderboard = sealed / "results.json"
    original = leaderboard.read_text()
    leaderboard.write_text("{ damaged")
    config = replace(_config(code_root), candidate_runners=_TWO_RUNNERS)
    run = ["run-candidate", "--experiment", str(sealed), "--candidate", "candidate-b"]
    assert run_runner_main(config, run) == 1
    assert "experiment rebuild-results" in capsys.readouterr().err

    assert _rebuild(sealed) == 0
    assert "was invalid" in capsys.readouterr().out
    assert leaderboard.read_text() == original
    assert _rebuild(sealed) == 0
    assert "already matches 2" in capsys.readouterr().out
    assert run_runner_main(config, run) == 0


def test_rebuild_reports_and_refuses_changes_without_a_reason(
    sealed: Path, code_root: Path, capsys
) -> None:
    _two_candidates(sealed, code_root)
    kept, dropped = json.loads((sealed / "results.json").read_text())
    forged = {**kept, "candidate_id": "hand-written"}
    leaderboard = sealed / "results.json"
    leaderboard.write_text(json.dumps([forged]))
    before = leaderboard.read_text()
    assert _rebuild(sealed) == 1
    err = capsys.readouterr().err
    assert "candidate_id: hand-written ->" in err
    assert f"Added {dropped['candidate_id']}" in err
    assert leaderboard.read_text() == before

    assert main(["rebuild-results", str(sealed), "--allow-changes"]) == 1
    capsys.readouterr()
    flags = ["--allow-changes", "--reason", "The leaderboard was hand-edited."]
    assert main(["rebuild-results", str(sealed), *flags]) == 0
    assert "Rebuilt" in capsys.readouterr().out
    assert main(["leaderboard", str(sealed)]) == 0
    assert "Exploratory" in capsys.readouterr().out


def test_rebuild_refuses_an_invalid_run_record(sealed: Path, code_root: Path, capsys) -> None:
    _two_candidates(sealed, code_root)
    leaderboard = sealed / "results.json"
    leaderboard.write_text("{ damaged")
    record = next((sealed / "work/runs").glob("*.json"))
    record.write_text("{ also damaged")
    assert _rebuild(sealed) == 1
    assert f"{record}: Run record is invalid" in capsys.readouterr().err
    assert leaderboard.read_text() == "{ damaged"


def test_rebuild_refuses_a_run_record_under_another_name(
    sealed: Path, code_root: Path, capsys
) -> None:
    _two_candidates(sealed, code_root)
    record = next((sealed / "work/runs").glob("*.json"))
    record.rename(record.with_name("00000000-0000-4000-8000-000000000000.json"))
    assert _rebuild(sealed) == 1
    assert "names another run" in capsys.readouterr().err


def test_rebuild_after_freeze_keeps_or_refuses_the_frozen_leaderboard(
    sealed: Path, code_root: Path, capsys
) -> None:
    _two_candidates(sealed, code_root)
    freeze_candidate(sealed, "candidate-a", "label")
    leaderboard = sealed / "results.json"
    original = leaderboard.read_text()
    leaderboard.write_text("{ damaged")
    assert _rebuild(sealed) == 0
    assert leaderboard.read_text() == original

    latest = json.loads(original)[0]["run_id"]
    (sealed / "work/runs" / f"{latest}.json").unlink()
    leaderboard.write_text("{ damaged")
    capsys.readouterr()
    assert _rebuild(sealed) == 1
    assert "Selection is frozen" in capsys.readouterr().err
    assert leaderboard.read_text() == "{ damaged"
