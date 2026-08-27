from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_crew.state import get_data_store, reset_data_store


@pytest.fixture(scope="session", autouse=True)
def _cheap_catboost_for_tests():
    """Cut CatBoost's boosting iterations for the whole test session.

    Measured on this suite: at the fixtures' n=200, a single CatBoost fit takes
    ~0.92s against ~0.06s for every other candidate combined -- roughly 85% of
    all model-fit time in one estimator. Since model selection, HPO, ensembling
    and explainability each fit it repeatedly (cross_val_predict alone is
    folds x members), that one default dominates the suite's runtime.

    300 iterations is the right *production* default and stays that way; no
    test here asserts anything about convergence quality, only that leaderboards
    sort, ensembles build, and reports come back well-formed -- all of which 30
    iterations demonstrates identically. Shrinking the fixtures instead would be
    the wrong lever: at n=200 per-fit overhead already dominates row count (4x
    less data buys ~1.9x, 10x fewer iterations buys ~11x), and those fixtures
    deliberately carry nulls, a constant column, a high-cardinality column, a
    datetime and string labels, which is what makes the cleaning and feature
    stages genuinely exercised.
    """
    from ds_crew.tools.model_tools import BASE_MODEL_KWARGS

    original = BASE_MODEL_KWARGS["catboost"]["iterations"]
    BASE_MODEL_KWARGS["catboost"]["iterations"] = 30
    yield
    BASE_MODEL_KWARGS["catboost"]["iterations"] = original


@pytest.fixture(scope="session", autouse=True)
def _isolated_mlflow_tracking(tmp_path_factory):
    """Point the whole test session at a throwaway MLflow tracking store.

    `POST /runs` now opens a real MLflow run via `logging_tools.start_mlflow_run`
    (see its docstring), so every test that creates a run through the service
    -- not just ones that mention MLflow -- would otherwise write against
    whatever MLFLOW_TRACKING_URI a developer's .env points at, either polluting
    a real mlflow.db or silently no-op-ing on a locked one.
    """
    import mlflow

    tracking_dir = tmp_path_factory.mktemp("mlflow_tracking")
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'mlflow.db'}")


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
