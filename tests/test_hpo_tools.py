from __future__ import annotations

import json

import optuna
import pytest
from _helpers import prepare_features

from ds_crew import settings
from ds_crew.tools.hpo_tools import TuneModelsTool, _suggest_params, run_optuna_study
from ds_crew.tools.model_tools import CANDIDATE_MODELS, TrainCandidateModelsTool


def _prepare_run(run_id, df, target):
    prepare_features(run_id, df, target)
    TrainCandidateModelsTool(run_id=run_id)._run(cv_folds=3)


def test_run_optuna_study_returns_result(classification_run, run_id, classification_df):
    _prepare_run(run_id, classification_df, "target")
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
    _prepare_run(run_id, classification_df, "target")
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
    _prepare_run(run_id, classification_df, "target")
    tool = TuneModelsTool(run_id=run_id, top_k=4)
    result = json.loads(
        tool._run(model_names=["logistic_regression"], n_trials=100000, timeout_s=100000)
    )
    assert result["results"][0]["n_trials"] <= 2


def test_tune_models_tool_end_to_end_updates_state(classification_run, run_id, classification_df):
    _prepare_run(run_id, classification_df, "target")
    best_name = classification_run.leaderboard.candidates[0].model_name
    tool = TuneModelsTool(run_id=run_id, top_k=4)
    result = json.loads(tool._run(model_names=[best_name], n_trials=3, timeout_s=30))
    assert result["run_id"] == run_id
    assert best_name in classification_run.hpo_results
    assert classification_run.history[-1]["stage"] == "hpo"


def test_tune_models_tool_isolates_one_failing_model(
    classification_run, run_id, classification_df, monkeypatch
):
    # A model with no defined search space raises ValueError on every trial --
    # this must be recorded as a warning and NOT prevent the other, valid model
    # in the same call from returning results.
    _prepare_run(run_id, classification_df, "target")
    top_two = [c.model_name for c in classification_run.leaderboard.candidates[:2]]
    good_name, bad_name = top_two

    import ds_crew.tools.hpo_tools as hpo_tools_module

    original = hpo_tools_module._suggest_params

    def _flaky_suggest_params(trial, model_name, n_fold_train_samples):
        if model_name == bad_name:
            raise ValueError("simulated bad search space")
        return original(trial, model_name, n_fold_train_samples)

    monkeypatch.setattr(hpo_tools_module, "_suggest_params", _flaky_suggest_params)

    tool = TuneModelsTool(run_id=run_id, top_k=2)
    result = json.loads(tool._run(model_names=top_two, n_trials=3, timeout_s=30))

    assert any(r["model_name"] == good_name for r in result["results"])
    assert any(bad_name in w for w in result["warnings"])


def test_suggest_params_covers_every_candidate_model():
    """Every model in CANDIDATE_MODELS must have a matching _suggest_params
    branch -- guards the exact failure mode the module's docstring calls out:
    a candidate with no defined search space raising ValueError at HPO time.
    """
    trial = optuna.create_study().ask()
    for task_models in CANDIDATE_MODELS.values():
        for name in task_models:
            params = _suggest_params(trial, name, n_fold_train_samples=100)
            assert isinstance(params, dict)


def test_suggest_params_bounds_knn_neighbors_to_fold_size():
    trial = optuna.create_study().ask()
    params = _suggest_params(trial, "knn", n_fold_train_samples=5)
    assert params["n_neighbors"] <= 4


def test_run_optuna_study_raises_when_every_trial_fails(
    classification_run, run_id, classification_df, monkeypatch
):
    # Every trial's suggested params raise -- catch=(Exception,) in
    # run_optuna_study means Optuna records each as a FAILed trial rather than
    # aborting the study outright, but with zero COMPLETE trials there's no
    # best_params/best_score to report, so the function must surface a clear
    # RuntimeError instead of letting Optuna's own ValueError leak out raw.
    import ds_crew.tools.hpo_tools as hpo_tools_module

    def _always_fail(trial, model_name, n_fold_train_samples):
        raise ValueError("simulated: no viable search space")

    monkeypatch.setattr(hpo_tools_module, "_suggest_params", _always_fail)

    _prepare_run(run_id, classification_df, "target")
    with pytest.raises(RuntimeError):
        run_optuna_study(
            classification_run.X_train,
            classification_run.y_train,
            "knn",
            "classification",
            n_trials=2,
            timeout_s=30,
            seed=42,
        )
