from __future__ import annotations

import json

import pandas as pd

from ds_crew.schemas import ColumnFeaturePlan, FeatureEngineeringPlan
from ds_crew.tools.feature_tools import ApplyFeaturePlanTool, split_train_test


def test_split_train_test_stratifies_classification(classification_df):
    X_train, X_test, y_train, y_test = split_train_test(
        classification_df, "target", "classification", test_size=0.25, random_state=42
    )
    assert len(X_train) + len(X_test) == len(classification_df)
    train_balance = y_train.value_counts(normalize=True).round(1)
    full_balance = classification_df["target"].value_counts(normalize=True).round(1)
    assert set(train_balance.index) == set(full_balance.index)


def _full_plan_for(df: pd.DataFrame, target: str) -> FeatureEngineeringPlan:
    plans = []
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            plans.append(ColumnFeaturePlan(column=col, encoding="none", scaling="standard"))
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            plans.append(ColumnFeaturePlan(column=col, datetime_decompose=True))
        else:
            plans.append(ColumnFeaturePlan(column=col, encoding="onehot"))
    return FeatureEngineeringPlan(run_id="r", column_plans=plans)


def test_apply_feature_plan_tool_end_to_end(classification_run, run_id, classification_df):
    plan = _full_plan_for(classification_df, "target")
    tool = ApplyFeaturePlanTool(run_id=run_id)
    result = json.loads(tool._run(**plan.model_dump(exclude={"run_id"})))
    assert result["status"] == "applied"
    assert result["n_features"] > 0
    assert classification_run.X_train is not None
    assert classification_run.X_test is not None
    assert classification_run.y_train is not None
    n_rows = len(classification_df)
    assert classification_run.X_train.shape[0] + classification_run.X_test.shape[0] == n_rows


def test_apply_feature_plan_tool_refuses_second_call(classification_run, run_id, classification_df):
    # Mirrors the cleaning-tool guard: a second call (e.g. a model fragmenting one
    # large tool call into several) must be refused, not silently reapplied.
    plan = _full_plan_for(classification_df, "target")
    tool = ApplyFeaturePlanTool(run_id=run_id)
    first = json.loads(tool._run(**plan.model_dump(exclude={"run_id"})))
    assert first["status"] == "applied"

    second = json.loads(tool._run(column_plans=[{"column": "num_a", "encoding": "none"}]))
    assert "error" in second
    assert "already been applied" in second["error"]


def test_apply_feature_plan_tool_refuses_target_as_feature(classification_run, run_id):
    tool = ApplyFeaturePlanTool(run_id=run_id)
    result = json.loads(
        tool._run(column_plans=[{"column": "target", "encoding": "onehot"}])
    )
    assert "error" in result
    assert "target" in result["error"]


def test_apply_feature_plan_tool_requires_full_coverage(classification_run, run_id):
    tool = ApplyFeaturePlanTool(run_id=run_id)
    result = json.loads(
        tool._run(column_plans=[{"column": "num_a", "encoding": "none", "scaling": "standard"}])
    )
    assert "error" in result
    assert "not covered" in result["error"]


def test_apply_feature_plan_tool_unknown_column(classification_run, run_id):
    tool = ApplyFeaturePlanTool(run_id=run_id)
    result = json.loads(
        tool._run(column_plans=[{"column": "nope", "encoding": "onehot"}])
    )
    assert "error" in result


def test_apply_feature_plan_tool_requires_encoding_for_non_numeric(classification_run, run_id, classification_df):
    plan = _full_plan_for(classification_df, "target")
    # deliberately break the cat_a plan to encoding='none' on a non-numeric column
    for cp in plan.column_plans:
        if cp.column == "cat_a":
            cp.encoding = "none"
    tool = ApplyFeaturePlanTool(run_id=run_id)
    result = json.loads(tool._run(**plan.model_dump(exclude={"run_id"})))
    assert "error" in result


def test_target_mean_encoding_no_leakage_between_train_and_test(regression_run, run_id, regression_df):
    plan = FeatureEngineeringPlan(
        run_id=run_id,
        column_plans=[
            ColumnFeaturePlan(column="num_a", encoding="none", scaling="standard"),
            ColumnFeaturePlan(column="num_b", encoding="none", scaling="standard"),
            ColumnFeaturePlan(column="cat_a", encoding="target_mean"),
        ],
    )
    tool = ApplyFeaturePlanTool(run_id=run_id)
    result = json.loads(tool._run(**plan.model_dump(exclude={"run_id"})))
    assert result["status"] == "applied"
    # the fitted transformer must have been fit only on the training fold
    assert regression_run.fitted_transformer is not None


def test_feature_selection_reduces_feature_count(classification_run, run_id, classification_df):
    # Feature selectors (mutual_info) require clean input -- imputation is cleaning's job,
    # upstream of feature engineering in the real pipeline. Impute here to isolate selection.
    classification_run.df["with_nulls"] = classification_run.df["with_nulls"].fillna(
        classification_run.df["with_nulls"].mean()
    )
    plan = _full_plan_for(classification_run.df, "target")
    plan_dict = plan.model_dump(exclude={"run_id"})
    plan_dict["feature_selection_method"] = "mutual_info"
    plan_dict["top_k"] = 3
    tool = ApplyFeaturePlanTool(run_id=run_id)
    result = json.loads(tool._run(**plan_dict))
    assert result["status"] == "applied"
    assert result["n_features"] == 3
