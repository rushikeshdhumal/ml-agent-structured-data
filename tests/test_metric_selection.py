from __future__ import annotations

import json

import pytest
from _helpers import prepare_features

from ds_crew.tools.hpo_tools import run_optuna_study
from ds_crew.tools.model_tools import (
    ALLOWED_METRICS,
    SetMetricTool,
    TrainCandidateModelsTool,
    resolve_cv_scorer,
)


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


def test_set_metric_tool_normalizes_separator_variants(classification_run, run_id):
    """Same normalization the plan enums use, for the same reason: "f1-macro"
    is a spelling of an allowed metric, not a different request.
    """
    tool = SetMetricTool(run_id=run_id)
    result = json.loads(tool._run(metric="f1-macro"))
    assert result["status"] == "set"
    assert classification_run.metric_name == "f1_macro"


def test_set_metric_tool_still_rejects_a_genuinely_unknown_metric(classification_run, run_id):
    tool = SetMetricTool(run_id=run_id)
    assert "error" in json.loads(tool._run(metric="f_one"))
    assert classification_run.metric_name is None


def test_set_metric_tool_rejects_disallowed_metric_for_task_type(classification_run, run_id):
    tool = SetMetricTool(run_id=run_id)
    result = json.loads(tool._run(metric="rmse"))
    assert "error" in result
    assert classification_run.metric_name is None


def test_set_metric_tool_rejects_non_r2_for_regression(regression_run, run_id):
    tool = SetMetricTool(run_id=run_id)
    result = json.loads(tool._run(metric="accuracy"))
    assert "error" in result


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
