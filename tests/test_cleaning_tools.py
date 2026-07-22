from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ds_crew.schemas import CleaningPlan, ColumnCleaningAction
from ds_crew.tools.cleaning_tools import (
    ApplyCleaningPlanTool,
    apply_structural_cleaning,
    fit_transform_cleaning,
    split_df,
)


def test_split_df_stratifies_classification(classification_df):
    df_train, df_test = split_df(
        classification_df, "target", "classification", test_size=0.25, random_state=42
    )
    assert len(df_train) + len(df_test) == len(classification_df)
    train_balance = df_train["target"].value_counts(normalize=True).round(1)
    full_balance = classification_df["target"].value_counts(normalize=True).round(1)
    assert set(train_balance.index) == set(full_balance.index)


def test_apply_structural_cleaning_drops_columns_and_dupes():
    df = pd.DataFrame({"x": [1, 1, 2], "drop_me": ["a", "a", "b"]})
    plan = CleaningPlan(run_id="r", drop_duplicate_rows=True, columns_to_drop=["drop_me"])
    out = apply_structural_cleaning(df, plan)
    assert "drop_me" not in out.columns
    assert len(out) == 2


def test_fit_transform_cleaning_mean_imputation_fits_on_train_only():
    # Train's mean (2.0) and test's mean (100.0) are wildly different -- a
    # correct fit-on-train implementation must fill test's gap with TRAIN's
    # mean, not test's own, proving no leakage of test values into the fit.
    df_train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    df_test = pd.DataFrame({"x": [None, 100.0, 100.0]})
    plan = CleaningPlan(
        run_id="r", actions=[ColumnCleaningAction(column="x", missing_strategy="mean")]
    )
    out_train, out_test = fit_transform_cleaning(df_train, df_test, plan)
    assert out_train["x"].isna().sum() == 0
    assert out_test["x"].iloc[0] == pytest.approx(2.0)


def test_fit_transform_cleaning_drop_rows_is_per_split():
    df_train = pd.DataFrame({"x": [1.0, None, 3.0]})
    df_test = pd.DataFrame({"x": [None, 5.0]})
    plan = CleaningPlan(
        run_id="r", actions=[ColumnCleaningAction(column="x", missing_strategy="drop_rows")]
    )
    out_train, out_test = fit_transform_cleaning(df_train, df_test, plan)
    assert len(out_train) == 2
    assert len(out_test) == 1


def test_fit_transform_cleaning_constant_fill_requires_value():
    df_train = pd.DataFrame({"x": [1.0, None]})
    df_test = pd.DataFrame({"x": [None]})
    plan = CleaningPlan(
        run_id="r", actions=[ColumnCleaningAction(column="x", missing_strategy="constant")]
    )
    with pytest.raises(ValueError):
        fit_transform_cleaning(df_train, df_test, plan)


def test_fit_transform_cleaning_iqr_bounds_from_train_applied_to_test():
    # Train has no outliers (bounds computed from train stay tight); a value
    # that's an outlier by train's bounds must get clipped in test too, even
    # though it wouldn't look like an outlier by test's own (tiny) sample.
    df_train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    df_test = pd.DataFrame({"x": [1000.0]})
    plan = CleaningPlan(
        run_id="r", actions=[ColumnCleaningAction(column="x", outlier_strategy="iqr_clip")]
    )
    out_train, out_test = fit_transform_cleaning(df_train, df_test, plan)
    assert out_test["x"].iloc[0] < 1000.0


def test_fit_transform_cleaning_knn_imputer_fit_on_train_transforms_test():
    df_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    df_test = pd.DataFrame({"a": [None], "b": [3.5]})
    plan = CleaningPlan(
        run_id="r", actions=[ColumnCleaningAction(column="a", missing_strategy="knn")]
    )
    out_train, out_test = fit_transform_cleaning(df_train, df_test, plan)
    assert out_train["a"].isna().sum() == 0
    assert out_test["a"].isna().sum() == 0


def test_mean_strategy_on_non_numeric_raises():
    df_train = pd.DataFrame({"x": ["a", None, "c"]})
    df_test = pd.DataFrame({"x": ["b"]})
    plan = CleaningPlan(run_id="r", actions=[ColumnCleaningAction(column="x", missing_strategy="mean")])
    with pytest.raises(ValueError):
        fit_transform_cleaning(df_train, df_test, plan)


def test_apply_cleaning_plan_tool_rejects_unknown_column(classification_run, run_id):
    tool = ApplyCleaningPlanTool(run_id=run_id)
    result = json.loads(
        tool._run(actions=[{"column": "does_not_exist", "missing_strategy": "mean"}])
    )
    assert "error" in result
    assert "does_not_exist" in result["error"]


def test_apply_cleaning_plan_tool_refuses_target_drop(classification_run, run_id):
    tool = ApplyCleaningPlanTool(run_id=run_id)
    result = json.loads(tool._run(columns_to_drop=["target"]))
    assert "error" in result
    assert "target" in result["error"]


def test_apply_cleaning_plan_tool_refuses_target_action(classification_run, run_id):
    tool = ApplyCleaningPlanTool(run_id=run_id)
    result = json.loads(
        tool._run(actions=[{"column": "target", "missing_strategy": "mode"}])
    )
    assert "error" in result


def test_apply_cleaning_plan_tool_applies_split_and_mutates_state(classification_run, run_id):
    tool = ApplyCleaningPlanTool(run_id=run_id)
    n_rows = len(classification_run.df)
    result = json.loads(
        tool._run(
            actions=[{"column": "with_nulls", "missing_strategy": "median"}],
        )
    )
    assert result["status"] == "applied"
    assert classification_run.df_train is not None
    assert classification_run.df_test is not None
    assert classification_run.df_train.shape[0] + classification_run.df_test.shape[0] == n_rows
    assert classification_run.df_train["with_nulls"].isna().sum() == 0
    assert classification_run.df_test["with_nulls"].isna().sum() == 0
    # state.df itself is never mutated by cleaning -- it stays the original raw data.
    assert classification_run.df.shape[0] == n_rows
    assert classification_run.history[-1]["stage"] == "cleaning"


def test_apply_cleaning_plan_tool_rejects_column_both_cleaned_and_dropped(
    classification_run, run_id
):
    # apply_structural_cleaning drops columns_to_drop before fit_transform_cleaning
    # runs the per-column actions -- without this check, a plan naming the same
    # column in both would pass validation and then hit a KeyError once cleaning
    # tries to access a column that's already gone.
    tool = ApplyCleaningPlanTool(run_id=run_id)
    result = json.loads(
        tool._run(
            actions=[{"column": "with_nulls", "missing_strategy": "mean"}],
            columns_to_drop=["with_nulls"],
        )
    )
    assert "error" in result
    assert "with_nulls" in result["error"]
    assert classification_run.cleaning_applied is False


def test_apply_cleaning_plan_tool_refuses_second_call(classification_run, run_id):
    # Some models fragment one large structured tool call into several partial
    # invocations; since every CleaningPlan field has a default, a second call
    # (even a harmless-looking fragment) must be refused, not silently reapplied.
    tool = ApplyCleaningPlanTool(run_id=run_id)
    first = json.loads(tool._run(actions=[{"column": "with_nulls", "missing_strategy": "median"}]))
    assert first["status"] == "applied"

    second = json.loads(tool._run(drop_duplicate_rows=True))
    assert "error" in second
    assert "already been applied" in second["error"]


def test_apply_cleaning_plan_tool_bad_strategy_returns_error_not_exception(
    classification_run, run_id
):
    tool = ApplyCleaningPlanTool(run_id=run_id)
    result = json.loads(
        tool._run(actions=[{"column": "cat_a", "missing_strategy": "mean"}])
    )
    assert "error" in result


def test_apply_cleaning_plan_tool_rejects_rare_class_split_gracefully(run_id):
    from ds_crew.state import get_data_store

    df = pd.DataFrame(
        {
            "x": np.arange(10, dtype=float),
            "target": ["yes"] * 9 + ["no"],  # "no" has only 1 member -- can't stratify-split
        }
    )
    state = get_data_store().create_run(run_id, df, target="target")
    state.task_type = "classification"

    tool = ApplyCleaningPlanTool(run_id=run_id)
    result = json.loads(tool._run())
    assert "error" in result
