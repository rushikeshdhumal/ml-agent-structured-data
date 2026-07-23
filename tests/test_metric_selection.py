from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from _helpers import prepare_features

from ds_crew.guardrails import validate_metric_choice_guardrail
from ds_crew.schemas import MetricChoice
from ds_crew.tools.hpo_tools import run_optuna_study
from ds_crew.tools.model_tools import (
    ALLOWED_METRICS,
    SetMetricTool,
    TrainCandidateModelsTool,
    resolve_cv_scorer,
)


def _task_output(pydantic_obj):
    return SimpleNamespace(pydantic=pydantic_obj)


def _prepare_run(run_id, df, target):
    prepare_features(run_id, df, target)


def test_resolve_cv_scorer_passes_through_non_roc_auc_metrics():
    import pandas as pd

    assert resolve_cv_scorer("f1_macro", pd.Series(["a", "b"])) == "f1_macro"
    assert resolve_cv_scorer("r2", pd.Series([1.0, 2.0])) == "r2"


def test_resolve_cv_scorer_uses_ovr_for_multiclass_roc_auc():
    import pandas as pd

    assert resolve_cv_scorer("roc_auc", pd.Series(["a", "b"])) == "roc_auc"
    assert resolve_cv_scorer("roc_auc", pd.Series(["a", "b", "c"])) == "roc_auc_ovr"


def test_set_metric_tool_accepts_allowed_metric(classification_run, run_id):
    tool = SetMetricTool(run_id=run_id)
    result = json.loads(tool._run(metric="balanced_accuracy"))
    assert result["status"] == "set"
    assert classification_run.metric_name == "balanced_accuracy"
    assert classification_run.history[-1]["stage"] == "model_selection"


def test_set_metric_tool_rejects_disallowed_metric_for_task_type(classification_run, run_id):
    tool = SetMetricTool(run_id=run_id)
    result = json.loads(tool._run(metric="rmse"))
    assert "error" in result
    assert classification_run.metric_name is None


def test_set_metric_tool_rejects_non_r2_for_regression(regression_run, run_id):
    tool = SetMetricTool(run_id=run_id)
    result = json.loads(tool._run(metric="accuracy"))
    assert "error" in result


def test_validate_metric_choice_guardrail_passes_allowed_metric(classification_run, run_id):
    choice = MetricChoice(run_id=run_id, metric="roc_auc", rationale="binary target")
    ok, result = validate_metric_choice_guardrail(_task_output(choice))
    assert ok is True
    assert result is choice


def test_validate_metric_choice_guardrail_rejects_disallowed_metric(classification_run, run_id):
    choice = MetricChoice(run_id=run_id, metric="rmse")
    ok, error = validate_metric_choice_guardrail(_task_output(choice))
    assert ok is False
    assert "rmse" in error


def test_validate_metric_choice_guardrail_rejects_wrong_type():
    ok, error = validate_metric_choice_guardrail(_task_output("not a MetricChoice"))
    assert ok is False


def test_validate_metric_choice_guardrail_rejects_unknown_run():
    choice = MetricChoice(run_id="ghost-run", metric="f1_macro")
    ok, error = validate_metric_choice_guardrail(_task_output(choice))
    assert ok is False


def test_leaderboard_metric_name_follows_chosen_metric(classification_run, run_id, classification_df):
    _prepare_run(run_id, classification_df, "target")
    SetMetricTool(run_id=run_id)._run(metric="accuracy")
    tool = TrainCandidateModelsTool(run_id=run_id)
    result = json.loads(tool._run(cv_folds=3))
    assert result["metric_name"] == "accuracy"
    assert classification_run.leaderboard.metric_name == "accuracy"


def test_leaderboard_defaults_to_metric_by_task_when_unset(classification_run, run_id, classification_df):
    _prepare_run(run_id, classification_df, "target")
    assert classification_run.metric_name is None
    tool = TrainCandidateModelsTool(run_id=run_id)
    result = json.loads(tool._run(cv_folds=3))
    assert result["metric_name"] == "f1_macro"


def test_hpo_study_optimizes_the_chosen_metric(classification_run, run_id, classification_df):
    _prepare_run(run_id, classification_df, "target")
    result = run_optuna_study(
        classification_run.X_train,
        classification_run.y_train,
        "logistic_regression",
        "classification",
        n_trials=3,
        timeout_s=30,
        seed=42,
        metric="accuracy",
    )
    assert isinstance(result.best_score, float)


@pytest.mark.parametrize("metric", sorted(ALLOWED_METRICS["classification"]))
def test_every_allowed_classification_metric_is_a_valid_cv_scorer(
    classification_run, run_id, classification_df, metric
):
    _prepare_run(run_id, classification_df, "target")
    SetMetricTool(run_id=run_id)._run(metric=metric)
    tool = TrainCandidateModelsTool(run_id=run_id)
    result = json.loads(tool._run(cv_folds=3))
    assert "candidates" in result
    assert len(result["candidates"]) >= 1
