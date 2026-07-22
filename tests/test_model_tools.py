from __future__ import annotations

import json

from _helpers import prepare_features

from ds_crew.tools.model_tools import (
    BASE_MODEL_KWARGS,
    CANDIDATE_MODELS,
    TrainCandidateModelsTool,
    run_cross_validation,
)


def test_run_cross_validation_classification(classification_run, run_id, classification_df):
    prepare_features(run_id, classification_df, "target")
    leaderboard = run_cross_validation(
        classification_run.X_train, classification_run.y_train, "classification", cv_folds=3
    )
    assert leaderboard.task_type == "classification"
    assert {c.model_name for c in leaderboard.candidates} == set(
        CANDIDATE_MODELS["classification"]
    )
    scores = [c.cv_mean_score for c in leaderboard.candidates]
    assert scores == sorted(scores, reverse=True)


def test_run_cross_validation_regression(regression_run, run_id, regression_df):
    prepare_features(run_id, regression_df, "target")
    leaderboard = run_cross_validation(
        regression_run.X_train, regression_run.y_train, "regression", cv_folds=3
    )
    assert leaderboard.task_type == "regression"
    assert leaderboard.metric_name == "r2"
    assert {c.model_name for c in leaderboard.candidates} == set(CANDIDATE_MODELS["regression"])


def test_train_candidate_models_tool_requires_features_first(classification_run, run_id):
    tool = TrainCandidateModelsTool(run_id=run_id)
    result = json.loads(tool._run())
    assert "error" in result


def test_train_candidate_models_tool_end_to_end(classification_run, run_id, classification_df):
    prepare_features(run_id, classification_df, "target")
    tool = TrainCandidateModelsTool(run_id=run_id)
    result = json.loads(tool._run(cv_folds=3))
    assert result["run_id"] == run_id
    assert len(result["candidates"]) == len(CANDIDATE_MODELS["classification"])
    assert classification_run.leaderboard is not None
    assert classification_run.history[-1]["stage"] == "model_selection"


def test_all_candidate_models_instantiate():
    """Every (name, cls) pair must accept its BASE_MODEL_KWARGS -- catches
    constructor-kwarg mismatches (e.g. passing random_state to a model that
    doesn't accept it) at test time rather than at agent runtime.
    """
    for task_models in CANDIDATE_MODELS.values():
        for name, cls in task_models.items():
            cls(**BASE_MODEL_KWARGS.get(name, {}))
