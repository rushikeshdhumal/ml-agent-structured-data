from __future__ import annotations

import pytest

from ds_crew.state import DataStore, get_data_store


def test_singleton_identity():
    assert get_data_store() is get_data_store()
    assert DataStore() is get_data_store()


def test_create_and_get_run(classification_df, run_id):
    store = get_data_store()
    state = store.create_run(run_id, classification_df, target="target")
    assert store.get(run_id) is state
    assert state.df.shape == classification_df.shape
    assert state.artifacts_dir.exists()


def test_get_unknown_run_raises():
    store = get_data_store()
    with pytest.raises(KeyError):
        store.get("does-not-exist")


def test_record_appends_history(classification_run):
    classification_run.record("eda", "profile_computed", {"n_cols": 9})
    assert classification_run.history == [
        {"stage": "eda", "action": "profile_computed", "detail": {"n_cols": 9}}
    ]


def test_store_is_isolated_across_runs(classification_df, regression_df):
    store = get_data_store()
    a = store.create_run("run-a", classification_df, target="target")
    b = store.create_run("run-b", regression_df, target="target")
    assert store.get("run-a") is a
    assert store.get("run-b") is b
    assert a.df.shape != b.df.shape or not a.df.equals(b.df)
