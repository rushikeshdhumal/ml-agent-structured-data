from __future__ import annotations

import json

import numpy as np
import pytest
from _helpers import prepare_features

import ds_crew.tools.explain_tools as explain_tools_module
from ds_crew.tools.ensemble_tools import EnsembleModelsTool
from ds_crew.tools.eval_tools import EvaluateModelsTool
from ds_crew.tools.explain_tools import (
    ExplainModelsTool,
    build_attributions,
    shap_method_for,
    source_column_of,
)
from ds_crew.tools.model_tools import CANDIDATE_MODELS, TrainCandidateModelsTool


def _prepare_evaluated_run(run_id, df, target, model_names=None, metric=None):
    """Bring a run all the way to 'evaluated', which is the only state
    explain_models will accept. Mirrors the real pipeline order.
    """
    prepare_features(run_id, df, target)
    TrainCandidateModelsTool(run_id=run_id)._run(cv_folds=3)
    from ds_crew.state import get_data_store

    state = get_data_store().get(run_id)
    if metric:
        state.metric_name = metric
        state.leaderboard.metric_name = metric
    names = model_names or [state.leaderboard.candidates[0].model_name]
    EvaluateModelsTool(run_id=run_id)._run(model_names=names)
    return state, names


# ---------------------------------------------------------------------------
# Source-column rollup
# ---------------------------------------------------------------------------


def test_source_column_of_splits_on_first_separator():
    assert source_column_of("cat_a__cat_a_red") == "cat_a"
    assert source_column_of("num_a__num_a") == "num_a"
    # A column whose own name contains the separator must still roll up to the
    # transformer name, which is why the split is bounded to the first one.
    assert source_column_of("odd__name__x0_a") == "odd"
    assert source_column_of("bare") == "bare"


def test_column_importance_rolls_onehot_columns_back_to_source_and_normalizes():
    feature_names = ["cat_a__cat_a_r", "cat_a__cat_a_g", "num_a__num_a"]
    perm_mean = np.array([0.1, 0.1, 0.2])
    perm_std = np.zeros(3)
    _, column_importance, unused = build_attributions(
        feature_names, perm_mean, perm_std, None, top_k=10
    )
    assert set(column_importance) == {"cat_a", "num_a"}
    # The two one-hot slices sum into their source column, so cat_a and num_a
    # each carry half -- the point of the rollup.
    assert column_importance["cat_a"] == pytest.approx(0.5)
    assert column_importance["num_a"] == pytest.approx(0.5)
    assert sum(column_importance.values()) == pytest.approx(1.0)
    assert unused == []


def test_negative_permutation_importance_is_treated_as_unused_not_subtracted():
    # A negative score means shuffling the feature *improved* the metric, i.e.
    # noise. Letting it subtract would understate its source column.
    feature_names = ["cat_a__cat_a_r", "cat_a__cat_a_g", "num_a__num_a"]
    perm_mean = np.array([0.4, -0.2, 0.4])
    _, column_importance, unused = build_attributions(
        feature_names, perm_mean, np.zeros(3), None, top_k=10
    )
    assert column_importance["cat_a"] == pytest.approx(0.5)
    assert "cat_a__cat_a_g" in unused


# ---------------------------------------------------------------------------
# The catboost/multiclass segfault guard
# ---------------------------------------------------------------------------


def test_shap_is_skipped_for_multiclass_catboost():
    # shap 0.52.0's TreeExplainer SEGFAULTS (exit 139, not an exception) on a
    # multiclass CatBoostClassifier, which would kill the whole crew process.
    # This must be a pre-emptive guard -- no try/except can recover from it.
    assert shap_method_for("catboost", 3) is None
    assert shap_method_for("catboost", 2) == "shap_tree"
    assert shap_method_for("catboost", None) == "shap_tree"


def test_shap_still_allowed_for_other_multiclass_tree_models():
    # Verified empirically: xgboost/lightgbm multiclass both return cleanly.
    # The guard is a catboost limitation, not a blanket multiclass one.
    assert shap_method_for("xgboost", 3) == "shap_tree"
    assert shap_method_for("lightgbm", 3) == "shap_tree"


def test_models_without_a_cheap_exact_explainer_are_permutation_only():
    assert shap_method_for("knn", 2) is None
    assert shap_method_for("ensemble", 2) is None


# ---------------------------------------------------------------------------
# Tool preconditions
# ---------------------------------------------------------------------------


def test_refuses_before_evaluation_has_run(classification_run, run_id, classification_df):
    # The ordering guard is what makes "X_test is scored exactly once"
    # enforceable: explanation may only read X_test after scoring is locked in.
    prepare_features(run_id, classification_df, "target")
    TrainCandidateModelsTool(run_id=run_id)._run(cv_folds=3)
    result = json.loads(ExplainModelsTool(run_id=run_id)._run())
    assert "error" in result
    assert "evaluate_models first" in result["error"]
    assert classification_run.explanation_applied is False


def test_refuses_second_call(classification_run, run_id, classification_df):
    _prepare_evaluated_run(run_id, classification_df, "target")
    tool = ExplainModelsTool(run_id=run_id)
    assert "reports" in json.loads(tool._run())
    second = json.loads(tool._run())
    assert "error" in second
    assert "already been called" in second["error"]


def test_rejects_models_that_were_never_evaluated(
    classification_run, run_id, classification_df
):
    _prepare_evaluated_run(run_id, classification_df, "target")
    result = json.loads(ExplainModelsTool(run_id=run_id)._run(model_names=["knn", "xgboost"]))
    assert "error" in result
    assert "no evaluation report" in result["error"]


def test_applied_flag_is_set_before_x_test_is_read(
    classification_run, run_id, classification_df, monkeypatch
):
    # Same lesson as EvaluateModelsTool: a call that fails partway through has
    # still consumed the resource, so the flag must not wait for a clean finish.
    _prepare_evaluated_run(run_id, classification_df, "target")

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(explain_tools_module, "explain_model", _boom)
    with pytest.raises(RuntimeError):
        ExplainModelsTool(run_id=run_id)._run()
    assert classification_run.explanation_applied is True


# ---------------------------------------------------------------------------
# End-to-end report content
# ---------------------------------------------------------------------------


def test_explains_every_classification_registry_model(
    classification_run, run_id, classification_df
):
    names = sorted(CANDIDATE_MODELS["classification"])
    state, _ = _prepare_evaluated_run(run_id, classification_df, "target", model_names=names)
    result = json.loads(ExplainModelsTool(run_id=run_id)._run(model_names=names))

    assert {r["model_name"] for r in result["reports"]} == set(names)
    for report in result["reports"]:
        # Permutation importance is the floor: every model gets attributions,
        # including knn, which has neither feature_importances_ nor coef_.
        assert report["top_features"], report["model_name"]
        assert report["column_importance"], report["model_name"]
        assert report["n_rows_explained"] > 0
        assert report["explained_on"] == "test"
    assert set(state.explanation_reports) == set(names)
    assert state.history[-1]["stage"] == "explanation"


def test_ensemble_gets_real_attributions(classification_run, run_id, classification_df):
    # The gap this whole layer exists to close: a VotingClassifier has no
    # feature_importances_/coef_, so EvaluationReport.feature_importances is
    # null for it -- yet it is frequently the model recommended for sign-off.
    prepare_features(run_id, classification_df, "target")
    TrainCandidateModelsTool(run_id=run_id)._run(cv_folds=3)
    ensemble_result = json.loads(EnsembleModelsTool(run_id=run_id)._run(max_members=3))
    if ensemble_result.get("skipped"):
        pytest.skip("too few viable ensemble members on this fixture")

    EvaluateModelsTool(run_id=run_id)._run(model_names=["ensemble"])
    assert classification_run.evaluation_reports["ensemble"].feature_importances is None

    result = json.loads(ExplainModelsTool(run_id=run_id)._run(model_names=["ensemble"]))
    report = result["reports"][0]
    assert report["attribution_method"] == "permutation_only"
    assert report["top_features"]
    assert report["column_importance"]


def test_roc_auc_metric_path_with_string_labels(
    classification_run, run_id, classification_df
):
    # roc_auc + a string-labelled target + the custom XGBoost wrapper is the
    # exact combination that has broken twice in this project's history.
    _prepare_evaluated_run(
        run_id, classification_df, "target", model_names=["xgboost"], metric="roc_auc"
    )
    result = json.loads(ExplainModelsTool(run_id=run_id)._run(model_names=["xgboost"]))
    report = result["reports"][0]
    assert report["metric_name"] == "roc_auc"
    assert report["top_features"]


def test_shap_failure_degrades_to_permutation_without_raising(
    classification_run, run_id, classification_df, monkeypatch
):
    _prepare_evaluated_run(run_id, classification_df, "target", model_names=["xgboost"])

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated shap failure")

    monkeypatch.setattr(explain_tools_module, "compute_shap_values", _boom)
    result = json.loads(ExplainModelsTool(run_id=run_id)._run(model_names=["xgboost"]))
    report = result["reports"][0]
    assert report["attribution_method"] == "permutation_only"
    assert report["top_features"]
    assert any("SHAP unavailable" in w for w in report["warnings"])


def test_shap_path_produces_local_explanations_and_shap_values(
    classification_run, run_id, classification_df
):
    _prepare_evaluated_run(run_id, classification_df, "target", model_names=["xgboost"])
    result = json.loads(ExplainModelsTool(run_id=run_id)._run(model_names=["xgboost"]))
    report = result["reports"][0]
    assert report["attribution_method"] == "shap_tree"
    assert any(f["mean_abs_shap"] is not None for f in report["top_features"])
    assert report["local_explanations"]
    kinds = {le["kind"] for le in report["local_explanations"]}
    assert kinds <= {"confident_correct", "confident_wrong", "most_uncertain"}


def test_shap_survives_crewais_warnings_monkeypatch():
    # crewai/__init__.py replaces warnings.warn with a wrapper that takes no
    # **kwargs, so shap's warnings.warn(..., skip_file_prefixes=...) raises
    # TypeError. Because crewai is always imported in the real pipeline, this
    # silently degraded EVERY run to permutation-only while isolated testing
    # of shap looked fine -- assert the scoped restore actually works.
    import warnings as warnings_module

    import crewai  # noqa: F401 -- imported for its import-time monkeypatch

    with pytest.raises(TypeError):
        warnings_module.warn("x", UserWarning, skip_file_prefixes=("a",))

    with explain_tools_module._unpatched_warnings():
        warnings_module.warn("x", UserWarning, skip_file_prefixes=("a",))

    # ...and crewai's patch is put back afterwards.
    with pytest.raises(TypeError):
        warnings_module.warn("x", UserWarning, skip_file_prefixes=("a",))


def test_surrogate_is_fit_against_model_predictions(
    classification_run, run_id, classification_df
):
    _prepare_evaluated_run(run_id, classification_df, "target", model_names=["xgboost"])
    result = json.loads(ExplainModelsTool(run_id=run_id)._run(model_names=["xgboost"]))
    report = result["reports"][0]
    assert report["surrogate_rules"]
    assert 0.0 <= report["surrogate_fidelity"] <= 1.0


def test_regression_path(regression_run, run_id, regression_df):
    state, names = _prepare_evaluated_run(run_id, regression_df, "target")
    result = json.loads(ExplainModelsTool(run_id=run_id)._run())
    report = result["reports"][0]
    assert report["metric_name"] == "r2"
    assert report["top_features"]
    # regression_df's target is a near-noiseless linear function of num_a/num_b,
    # so those two must dominate -- a falsifiable check that the attributions
    # track real signal rather than just being well-formed.
    top_columns = list(report["column_importance"])[:2]
    assert {"num_a", "num_b"} == set(top_columns), report["column_importance"]


def test_noise_column_is_reported_as_unimportant(regression_run, run_id, regression_df):
    # cat_a is unrelated to regression_df's target by construction.
    _prepare_evaluated_run(run_id, regression_df, "target")
    result = json.loads(ExplainModelsTool(run_id=run_id)._run())
    column_importance = result["reports"][0]["column_importance"]
    assert column_importance.get("cat_a", 0.0) < column_importance["num_a"]
