from __future__ import annotations

import json

import pandas as pd
import pytest

from ds_crew.schemas import CleaningPlan, ColumnCleaningAction
from ds_crew.tools.cleaning_tools import ApplyCleaningPlanTool, apply_cleaning


def test_apply_cleaning_mean_imputation():
    df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0]})
    plan = CleaningPlan(
        run_id="r",
        actions=[ColumnCleaningAction(column="x", missing_strategy="mean")],
    )
    out = apply_cleaning(df, plan)
    assert out["x"].isna().sum() == 0
    assert out["x"].iloc[2] == pytest.approx((1.0 + 2.0 + 4.0) / 3)


def test_apply_cleaning_drop_rows():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    plan = CleaningPlan(
        run_id="r", actions=[ColumnCleaningAction(column="x", missing_strategy="drop_rows")]
    )
    out = apply_cleaning(df, plan)
    assert len(out) == 2


def test_apply_cleaning_constant_fill_requires_value():
    df = pd.DataFrame({"x": [1.0, None]})
    plan = CleaningPlan(
        run_id="r", actions=[ColumnCleaningAction(column="x", missing_strategy="constant")]
    )
    with pytest.raises(ValueError):
        apply_cleaning(df, plan)


def test_apply_cleaning_iqr_clip_outliers():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 1000]})
    plan = CleaningPlan(
        run_id="r", actions=[ColumnCleaningAction(column="x", outlier_strategy="iqr_clip")]
    )
    out = apply_cleaning(df, plan)
    assert out["x"].max() < 1000


def test_apply_cleaning_drops_columns_and_dupes():
    df = pd.DataFrame({"x": [1, 1, 2], "drop_me": ["a", "a", "b"]})
    plan = CleaningPlan(run_id="r", drop_duplicate_rows=True, columns_to_drop=["drop_me"])
    out = apply_cleaning(df, plan)
    assert "drop_me" not in out.columns
    assert len(out) == 2


def test_mean_strategy_on_non_numeric_raises():
    df = pd.DataFrame({"x": ["a", None, "c"]})
    plan = CleaningPlan(run_id="r", actions=[ColumnCleaningAction(column="x", missing_strategy="mean")])
    with pytest.raises(ValueError):
        apply_cleaning(df, plan)


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


def test_apply_cleaning_plan_tool_applies_and_mutates_state(classification_run, run_id):
    tool = ApplyCleaningPlanTool(run_id=run_id)
    before_shape = classification_run.df.shape
    result = json.loads(
        tool._run(
            actions=[{"column": "with_nulls", "missing_strategy": "median"}],
        )
    )
    assert result["status"] == "applied"
    assert classification_run.df.shape != before_shape or classification_run.df["with_nulls"].isna().sum() == 0
    assert classification_run.history[-1]["stage"] == "cleaning"


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
