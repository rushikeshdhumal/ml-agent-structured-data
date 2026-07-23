from __future__ import annotations

import json

from _helpers import prepare_features

from ds_crew import settings
from ds_crew.tools.ensemble_tools import EnsembleModelsTool
from ds_crew.tools.eval_tools import EvaluateModelsTool
from ds_crew.tools.model_tools import SetMetricTool, TrainCandidateModelsTool


def _prepare_run(run_id, df, target, metric=None):
    prepare_features(run_id, df, target)
    if metric:
        SetMetricTool(run_id=run_id)._run(metric=metric)
    TrainCandidateModelsTool(run_id=run_id)._run(cv_folds=3)


def test_build_ensemble_requires_leaderboard_first(classification_run, run_id):
    tool = EnsembleModelsTool(run_id=run_id)
    result = json.loads(tool._run())
    assert "error" in result


def test_build_ensemble_end_to_end_classification(
    classification_run, run_id, classification_df, monkeypatch
):
    monkeypatch.setattr(settings, "ENSEMBLE_WEIGHT_TRIALS", 5)
    monkeypatch.setattr(settings, "DEFAULT_CV_FOLDS", 3)
    _prepare_run(run_id, classification_df, "target")

    tool = EnsembleModelsTool(run_id=run_id)
    result = json.loads(tool._run(max_members=3))

    assert result["strategy"] in {"equal_voting", "weighted_voting", "greedy", "stacking"}
    assert len(result["member_models"]) <= 3
    assert "ensemble" in classification_run.fitted_models
    assert any(c.model_name == "ensemble" for c in classification_run.leaderboard.candidates)
    assert classification_run.ensemble_applied is True
    assert classification_run.ensemble_report is not None
    assert classification_run.history[-1]["stage"] == "ensemble"


def test_build_ensemble_respects_max_members_hard_cap(
    classification_run, run_id, classification_df, monkeypatch
):
    monkeypatch.setattr(settings, "ENSEMBLE_WEIGHT_TRIALS", 5)
    monkeypatch.setattr(settings, "DEFAULT_CV_FOLDS", 3)
    monkeypatch.setattr(settings, "MAX_ENSEMBLE_MEMBERS", 2)
    _prepare_run(run_id, classification_df, "target")

    tool = EnsembleModelsTool(run_id=run_id)
    # Request more members than the hard cap allows -- the tool-body min()
    # clamp must win regardless, mirroring TuneModelsTool's n_trials/timeout_s
    # double-layer cap pattern.
    result = json.loads(tool._run(max_members=10))
    assert len(result["member_models"]) <= 2


def test_build_ensemble_refuses_second_call(
    classification_run, run_id, classification_df, monkeypatch
):
    monkeypatch.setattr(settings, "ENSEMBLE_WEIGHT_TRIALS", 5)
    monkeypatch.setattr(settings, "DEFAULT_CV_FOLDS", 3)
    _prepare_run(run_id, classification_df, "target")

    tool = EnsembleModelsTool(run_id=run_id)
    first = json.loads(tool._run())
    assert "strategy" in first

    second = json.loads(tool._run())
    assert "error" in second
    assert "already been called" in second["error"]


def test_build_ensemble_regression(regression_run, run_id, regression_df, monkeypatch):
    monkeypatch.setattr(settings, "ENSEMBLE_WEIGHT_TRIALS", 5)
    monkeypatch.setattr(settings, "DEFAULT_CV_FOLDS", 3)
    _prepare_run(run_id, regression_df, "target")

    tool = EnsembleModelsTool(run_id=run_id)
    result = json.loads(tool._run())

    assert result["metric_name"] == "r2"
    assert "ensemble" in regression_run.fitted_models
    assert any(c.model_name == "ensemble" for c in regression_run.leaderboard.candidates)


def test_build_ensemble_skips_when_fewer_than_two_viable_members(
    classification_run, run_id, classification_df, monkeypatch
):
    monkeypatch.setattr(settings, "DEFAULT_CV_FOLDS", 3)
    _prepare_run(run_id, classification_df, "target")

    import ds_crew.tools.ensemble_tools as ensemble_tools_module

    original = ensemble_tools_module._build_estimator
    best_name = classification_run.leaderboard.candidates[0].model_name

    def _mostly_broken(name, task_type, params):
        if name != best_name:
            raise ValueError("simulated broken member")
        return original(name, task_type, params)

    monkeypatch.setattr(ensemble_tools_module, "_build_estimator", _mostly_broken)

    tool = EnsembleModelsTool(run_id=run_id)
    result = json.loads(tool._run())
    assert result.get("skipped") is True
    assert classification_run.ensemble_applied is False
    assert "ensemble" not in classification_run.fitted_models


def test_evaluate_models_scores_prefitted_ensemble_without_refitting(
    classification_run, run_id, classification_df, monkeypatch
):
    monkeypatch.setattr(settings, "ENSEMBLE_WEIGHT_TRIALS", 5)
    monkeypatch.setattr(settings, "DEFAULT_CV_FOLDS", 3)
    _prepare_run(run_id, classification_df, "target")
    EnsembleModelsTool(run_id=run_id)._run()
    ensemble_before = classification_run.fitted_models["ensemble"]

    tool = EvaluateModelsTool(run_id=run_id)
    result = json.loads(tool._run(model_names=["ensemble"]))

    report = result["reports"][0]
    assert report["model_name"] == "ensemble"
    assert "f1_macro" in report["metrics"]
    # evaluate_candidate must use the ensemble as-is (prefitted_model), not
    # reconstruct/refit it from CANDIDATE_MODELS -- "ensemble" isn't a
    # registry name, so refitting would KeyError before ever reaching scoring.
    assert classification_run.fitted_models["ensemble"] is ensemble_before
    assert classification_run.evaluation_reports["ensemble"] is not None


def test_ensemble_with_roc_auc_metric_and_xgboost_member_round_trips(
    classification_run, run_id, classification_df, monkeypatch
):
    # Known risk: _XGBClassifierWithLabelEncoding's own classes_ reflects the
    # internal integer encoding XGBoost's sklearn API requires at fit time,
    # not the original string labels -- eval_tools.py/ensemble_tools.py both
    # derive the canonical class order from y_train instead of model.classes_
    # specifically to handle this. Verify the full round trip (CV with
    # roc_auc -> ensemble -> evaluate, with xgboost as a leaderboard
    # candidate and a possible ensemble member) doesn't crash and produces a
    # valid roc_auc score for both a raw xgboost candidate and the ensemble.
    monkeypatch.setattr(settings, "ENSEMBLE_WEIGHT_TRIALS", 5)
    monkeypatch.setattr(settings, "DEFAULT_CV_FOLDS", 3)
    _prepare_run(run_id, classification_df, "target", metric="roc_auc")
    assert "xgboost" in {c.model_name for c in classification_run.leaderboard.candidates}

    EnsembleModelsTool(run_id=run_id)._run(max_members=5)

    result = json.loads(
        EvaluateModelsTool(run_id=run_id)._run(model_names=["xgboost", "ensemble"])
    )
    for report in result["reports"]:
        assert 0.0 <= report["metrics"]["roc_auc"] <= 1.0
