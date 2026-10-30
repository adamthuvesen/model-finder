from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from model_finder.final_evaluation import evaluate_final
from model_finder.holdout import (
    load_sealed_holdout,
    load_search_splits,
    read_holdout_seal,
    seal_holdout,
    sealed_file_path,
)
from model_finder.splits import HoldoutSplits, SearchSplits, temporal_split
from tests.conftest import fitted, rows


def _sealed_file(experiment: Path) -> Path:
    return sealed_file_path(read_holdout_seal(experiment))


def test_workspace_keeps_search_rows_and_only_a_hash_of_the_holdout(
    sealed: Path, isolated_holdout_store: Path
) -> None:
    sealed_file = _sealed_file(sealed)
    assert sealed_file.is_relative_to(isolated_holdout_store)
    holdout = pd.read_parquet(sealed_file)
    assert sealed_file.stat().st_mode & 0o777 == 0o600

    search = load_search_splits(sealed)
    assert len(search.train) + len(search.validation) + len(holdout) == len(rows())
    workspace_rows = pd.concat([search.train, search.validation])
    assert set(workspace_rows["when"]).isdisjoint(holdout["when"])

    record = json.loads((sealed / "holdout_seal.json").read_text())
    assert set(record) == {"schema_version", "experiment_uid", "holdout_sha256", "holdout_rows"}
    assert record["holdout_rows"] == len(holdout)
    workspace_text = "".join(
        path.read_text(errors="ignore") for path in sealed.rglob("*") if path.is_file()
    )
    assert str(sealed_file) not in workspace_text


def test_search_splits_have_no_holdout_attribute(sealed: Path) -> None:
    splits = load_search_splits(sealed)
    assert not hasattr(splits, "holdout")
    assert not hasattr(splits, "test")


def test_seal_defaults_to_local_share(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("XDG_DATA_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    search, holdout = temporal_split(rows(seed=3), "when")
    other = tmp_path / "other"
    other.mkdir()
    seal = seal_holdout(other, search, holdout)
    assert sealed_file_path(seal).is_relative_to(tmp_path / "home/.local/share/model-finder")


def test_resealing_is_refused(sealed: Path) -> None:
    search, holdout = temporal_split(rows(), "when")
    with pytest.raises(ValueError, match="already sealed"):
        seal_holdout(sealed, search, holdout)


def test_same_named_experiments_do_not_share_a_store(tmp_path: Path) -> None:
    first = tmp_path / "p" / "same"
    second = tmp_path / "q" / "same"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    seal_one = seal_holdout(first, *temporal_split(rows(seed=1), "when"))
    seal_two = seal_holdout(second, *temporal_split(rows(seed=2), "when"))
    assert seal_one.experiment_uid != seal_two.experiment_uid
    assert sealed_file_path(seal_one).is_file()
    assert sealed_file_path(seal_two).is_file()


def test_store_inside_the_experiment_is_refused(tmp_path: Path, monkeypatch) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(experiment / "store"))
    search, holdout = temporal_split(rows(), "when")
    with pytest.raises(ValueError, match="outside the experiment"):
        seal_holdout(experiment, search, holdout)


def test_empty_holdout_is_refused(tmp_path: Path) -> None:
    search, holdout = temporal_split(rows(), "when")
    empty = HoldoutSplits(holdout=holdout.holdout.iloc[:0])
    with pytest.raises(ValueError, match="empty"):
        seal_holdout(tmp_path, SearchSplits(search.train, search.validation), empty)


def test_load_rejects_a_changed_file(sealed: Path) -> None:
    seal = read_holdout_seal(sealed)
    pd.DataFrame({"signal": [1.0]}).to_parquet(_sealed_file(sealed))
    with pytest.raises(ValueError, match="changed after sealing"):
        load_sealed_holdout(seal)


def test_load_rejects_a_wrong_row_count(sealed: Path) -> None:
    seal = read_holdout_seal(sealed)
    wrong = seal.model_copy(update={"holdout_rows": seal.holdout_rows + 1})
    with pytest.raises(ValueError, match="row count"):
        load_sealed_holdout(wrong)


def test_evaluate_final_without_a_freeze_never_opens_the_store(scored: Path) -> None:
    _sealed_file(scored).unlink()
    with pytest.raises(ValueError, match="No frozen selection"):
        evaluate_final(scored, {"candidate-a": fitted})


def test_temporal_split_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="missing"):
        temporal_split(rows(), "absent")
    with pytest.raises(ValueError, match="too small"):
        temporal_split(rows(5), "when")
    with pytest.raises(ValueError, match="fractions"):
        temporal_split(rows(), "when", train_fraction=0.9, validation_fraction=0.1)


def test_temporal_split_holds_out_the_latest_rows() -> None:
    search, holdout = temporal_split(rows().sample(frac=1, random_state=1), "when")
    assert search.train["when"].max() < search.validation["when"].min()
    assert search.validation["when"].max() < holdout.holdout["when"].min()
