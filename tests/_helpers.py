"""Shared test setup that mirrors the real pipeline order: cleaning (which
also performs the train/test split) must run before feature engineering.
Used by tests that only care about model selection/HPO/evaluation and just
need a populated X_train/X_test/y_train/y_test to work with.
"""

from __future__ import annotations

import json

import pandas as pd

from ds_crew.schemas import CleaningPlan, ColumnCleaningAction, ColumnFeaturePlan, FeatureEngineeringPlan
from ds_crew.tools.cleaning_tools import ApplyCleaningPlanTool
from ds_crew.tools.feature_tools import ApplyFeaturePlanTool


def prepare_features(run_id: str, df: pd.DataFrame, target: str) -> None:
    cleaning_actions = []
    for col in df.select_dtypes(include="number").columns:
        if col != target and df[col].isna().any():
            cleaning_actions.append(ColumnCleaningAction(column=col, missing_strategy="mean"))
    cleaning_plan = CleaningPlan(run_id=run_id, actions=cleaning_actions)
    cleaning_result = json.loads(
        ApplyCleaningPlanTool(run_id=run_id)._run(**cleaning_plan.model_dump(exclude={"run_id"}))
    )
    assert cleaning_result["status"] == "applied", cleaning_result

    plans = []
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            plans.append(ColumnFeaturePlan(column=col, encoding="none", scaling="standard"))
        else:
            plans.append(ColumnFeaturePlan(column=col, encoding="onehot"))
    feature_plan = FeatureEngineeringPlan(run_id=run_id, column_plans=plans)
    feature_result = json.loads(
        ApplyFeaturePlanTool(run_id=run_id)._run(**feature_plan.model_dump(exclude={"run_id"}))
    )
    assert feature_result["status"] == "applied", feature_result
