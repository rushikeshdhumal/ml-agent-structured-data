from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_crew.state import get_data_store, reset_data_store


@pytest.fixture(autouse=True)
def _clean_data_store():
    reset_data_store()
    yield
    reset_data_store()


@pytest.fixture
def classification_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "num_a": rng.normal(0, 1, n),
            "num_b": rng.normal(5, 2, n).round(2),
            "cat_a": rng.choice(["red", "green", "blue"], n),
            "cat_b_high_card": [f"id_{i}" for i in range(n)],
            "bool_a": rng.choice([True, False], n),
            "date_a": pd.date_range("2023-01-01", periods=n, freq="D"),
            "with_nulls": [None if i % 10 == 0 else rng.normal() for i in range(n)],
            "constant_col": [1] * n,
            "target": rng.choice(["yes", "no"], n),
        }
    )


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "num_a": x1,
            "num_b": x2,
            "cat_a": rng.choice(["low", "mid", "high"], n),
            "target": 3 * x1 - 2 * x2 + rng.normal(0, 0.1, n),
        }
    )


@pytest.fixture
def run_id() -> str:
    return "test-run-001"


@pytest.fixture
def classification_run(classification_df, run_id):
    store = get_data_store()
    state = store.create_run(run_id, classification_df.copy(), target="target")
    state.task_type = "classification"
    return state


@pytest.fixture
def regression_run(regression_df, run_id):
    store = get_data_store()
    state = store.create_run(run_id, regression_df.copy(), target="target")
    state.task_type = "regression"
    return state
