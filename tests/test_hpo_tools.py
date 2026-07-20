from __future__ import annotations

import json

import optuna
import pandas as pd

from ds_crew import settings
from ds_crew.schemas import ColumnFeaturePlan, FeatureEngineeringPlan
from ds_crew.tools.feature_tools import ApplyFeaturePlanTool
from ds_crew.tools.hpo_tools import TuneModelsTool, _suggest_params, run_optuna_study
from ds_crew.tools.model_tools import CANDIDATE_MODELS, TrainCandidateModelsTool


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


def test_run_optuna_study_returns_result(classification_run, run_id, classification_df):
    _prepare_run(classification_run, run_id, classification_df, "target")
    best_name = classification_run.leaderboard.candidates[0].model_name
    result = run_optuna_study(
        classification_run.X_train,
        classification_run.y_train,
        best_name,
        "classification",
        n_trials=3,
        timeout_s=30,
        seed=42,
    )
    assert result.model_name == best_name
    assert result.n_trials == 3
    assert isinstance(result.best_params, dict)


def test_tune_models_tool_requires_leaderboard_first(classification_run, run_id):
    tool = TuneModelsTool(run_id=run_id)
    result = json.loads(tool._run(model_names=["random_forest"], n_trials=2, timeout_s=10))
    assert "error" in result


def test_tune_models_tool_rejects_model_outside_top_k(classification_run, run_id, classification_df):
    _prepare_run(classification_run, run_id, classification_df, "target")
    worst_name = classification_run.leaderboard.candidates[-1].model_name
    tool = TuneModelsTool(run_id=run_id, top_k=1)
    result = json.loads(tool._run(model_names=[worst_name], n_trials=2, timeout_s=10))
    assert "error" in result


def test_tune_models_tool_clamps_budget(classification_run, run_id, classification_df, monkeypatch):
    # Patch the caps low so this test stays fast while still proving the tool-body
    # min() clamp is enforced independently of the Pydantic Field bound (which is
    # bypassed here since we call _run() directly, not through tool-call validation).
    monkeypatch.setattr(settings, "MAX_HPO_TRIALS", 2)
    monkeypatch.setattr(settings, "MAX_HPO_TIMEOUT_S", 5)
    _prepare_run(classification_run, run_id, classification_df, "target")
    tool = TuneModelsTool(run_id=run_id, top_k=4)
    result = json.loads(
        tool._run(model_names=["logistic_regression"], n_trials=100000, timeout_s=100000)
    )
    assert result["results"][0]["n_trials"] <= 2


def test_tune_models_tool_end_to_end_updates_state(classification_run, run_id, classification_df):
    _prepare_run(classification_run, run_id, classification_df, "target")
    best_name = classification_run.leaderboard.candidates[0].model_name
    tool = TuneModelsTool(run_id=run_id, top_k=4)
    result = json.loads(tool._run(model_names=[best_name], n_trials=3, timeout_s=30))
    assert result["run_id"] == run_id
    assert best_name in classification_run.hpo_results
    assert classification_run.history[-1]["stage"] == "hpo"


def test_suggest_params_covers_every_candidate_model():
    """Every model in CANDIDATE_MODELS must have a matching _suggest_params
    branch -- guards the exact failure mode the module's docstring calls out:
    a candidate with no defined search space raising ValueError at HPO time.
    """
    trial = optuna.create_study().ask()
    for task_models in CANDIDATE_MODELS.values():
        for name in task_models:
            params = _suggest_params(trial, name)
            assert isinstance(params, dict)
