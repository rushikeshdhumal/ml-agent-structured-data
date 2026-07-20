from __future__ import annotations

import json

import pandas as pd

from ds_crew.schemas import ColumnFeaturePlan, FeatureEngineeringPlan
from ds_crew.tools.eval_tools import EvaluateModelsTool
from ds_crew.tools.feature_tools import ApplyFeaturePlanTool
from ds_crew.tools.model_tools import TrainCandidateModelsTool


def _prepare_run(state, run_id, df, target):
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    state.df = df
    plans = []
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            plans.append(ColumnFeaturePlan(column=col, encoding="none", scaling="standard"))
        else:
            plans.append(ColumnFeaturePlan(column=col, encoding="onehot"))
    plan = FeatureEngineeringPlan(run_id=run_id, column_plans=plans)
    ApplyFeaturePlanTool(run_id=run_id)._run(**plan.model_dump(exclude={"run_id"}))
    TrainCandidateModelsTool(run_id=run_id)._run(cv_folds=3)


def test_evaluate_models_tool_requires_leaderboard_first(classification_run, run_id):
    tool = EvaluateModelsTool(run_id=run_id)
    result = json.loads(tool._run(model_names=["random_forest"]))
    assert "error" in result


def test_evaluate_models_tool_rejects_unknown_model(classification_run, run_id, classification_df):
    _prepare_run(classification_run, run_id, classification_df, "target")
    tool = EvaluateModelsTool(run_id=run_id)
    result = json.loads(tool._run(model_names=["not_a_model"]))
    assert "error" in result


def test_evaluate_models_tool_classification_metrics(classification_run, run_id, classification_df):
    _prepare_run(classification_run, run_id, classification_df, "target")
    best_name = classification_run.leaderboard.candidates[0].model_name
    tool = EvaluateModelsTool(run_id=run_id)
    result = json.loads(tool._run(model_names=[best_name]))
    report = result["reports"][0]
    assert set(report["metrics"]) == {"accuracy", "f1_macro", "precision_macro", "recall_macro"}
    assert report["confusion_matrix"] is not None
    assert classification_run.fitted_models[best_name] is not None
    assert classification_run.history[-1]["stage"] == "evaluation"


def test_evaluate_models_tool_regression_metrics(regression_run, run_id, regression_df):
    _prepare_run(regression_run, run_id, regression_df, "target")
    best_name = regression_run.leaderboard.candidates[0].model_name
    tool = EvaluateModelsTool(run_id=run_id)
    result = json.loads(tool._run(model_names=[best_name]))
    report = result["reports"][0]
    assert set(report["metrics"]) == {"r2", "mae", "rmse"}
    assert report["confusion_matrix"] is None


def test_evaluate_models_tool_flags_near_perfect_as_leakage_suspicion(
    regression_run, run_id, regression_df
):
    # regression_df's target is an almost-noiseless linear function of num_a/num_b,
    # so a good regressor should score close to r2=1.0 and trip the leakage flag.
    _prepare_run(regression_run, run_id, regression_df, "target")
    best_name = regression_run.leaderboard.candidates[0].model_name
    tool = EvaluateModelsTool(run_id=run_id)
    result = json.loads(tool._run(model_names=[best_name]))
    report = result["reports"][0]
    assert report["leakage_suspicion"] is True
