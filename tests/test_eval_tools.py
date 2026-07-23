from __future__ import annotations

import json

import pytest
from _helpers import prepare_features

import ds_crew.tools.eval_tools as eval_tools_module
from ds_crew.tools.eval_tools import EvaluateModelsTool
from ds_crew.tools.model_tools import TrainCandidateModelsTool


def _prepare_run(run_id, df, target):
    prepare_features(run_id, df, target)
    TrainCandidateModelsTool(run_id=run_id)._run(cv_folds=3)


def test_evaluate_models_tool_requires_leaderboard_first(classification_run, run_id):
    tool = EvaluateModelsTool(run_id=run_id)
    result = json.loads(tool._run(model_names=["random_forest"]))
    assert "error" in result


def test_evaluate_models_tool_rejects_unknown_model(classification_run, run_id, classification_df):
    _prepare_run(run_id, classification_df, "target")
    tool = EvaluateModelsTool(run_id=run_id)
    result = json.loads(tool._run(model_names=["not_a_model"]))
    assert "error" in result


def test_evaluate_models_tool_classification_metrics(classification_run, run_id, classification_df):
    _prepare_run(run_id, classification_df, "target")
    best_name = classification_run.leaderboard.candidates[0].model_name
    tool = EvaluateModelsTool(run_id=run_id)
    result = json.loads(tool._run(model_names=[best_name]))
    report = result["reports"][0]
    assert {"accuracy", "f1_macro", "precision_macro", "recall_macro", "balanced_accuracy"} <= set(
        report["metrics"]
    )
    assert report["confusion_matrix"] is not None
    assert classification_run.fitted_models[best_name] is not None
    assert classification_run.history[-1]["stage"] == "evaluation"


def test_evaluate_models_tool_regression_metrics(regression_run, run_id, regression_df):
    _prepare_run(run_id, regression_df, "target")
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
    _prepare_run(run_id, regression_df, "target")
    best_name = regression_run.leaderboard.candidates[0].model_name
    tool = EvaluateModelsTool(run_id=run_id)
    result = json.loads(tool._run(model_names=[best_name]))
    report = result["reports"][0]
    assert report["leakage_suspicion"] is True


def test_evaluate_models_tool_refuses_second_call(classification_run, run_id, classification_df):
    # The held-out test set is only a valid final check if it's scored once --
    # repeated calls (even with a different model_names subset) must be refused.
    _prepare_run(run_id, classification_df, "target")
    best_name = classification_run.leaderboard.candidates[0].model_name
    tool = EvaluateModelsTool(run_id=run_id)
    first = json.loads(tool._run(model_names=[best_name]))
    assert "reports" in first

    second = json.loads(tool._run(model_names=[best_name]))
    assert "error" in second
    assert "already been called" in second["error"]


def test_evaluate_models_tool_sets_applied_flag_before_scoring_starts(
    classification_run, run_id, classification_df, monkeypatch
):
    # evaluation_applied must be set before the scoring loop runs, not after
    # it completes -- otherwise a mid-batch failure (X_test already used for
    # whichever models ran before the failing one) would leave the flag
    # False, and a "retry" would be allowed to touch X_test again.
    _prepare_run(run_id, classification_df, "target")
    names = [c.model_name for c in classification_run.leaderboard.candidates[:2]]

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure mid-batch")

    monkeypatch.setattr(eval_tools_module, "evaluate_candidate", _boom)

    tool = EvaluateModelsTool(run_id=run_id)
    with pytest.raises(RuntimeError):
        tool._run(model_names=names)

    assert classification_run.evaluation_applied is True

    retry = json.loads(tool._run(model_names=names))
    assert "error" in retry
    assert "already been called" in retry["error"]
