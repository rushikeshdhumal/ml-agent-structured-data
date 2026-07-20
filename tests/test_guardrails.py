from __future__ import annotations

from types import SimpleNamespace

from ds_crew.guardrails import (
    make_finalize_called_guardrail,
    prevent_target_leakage_guardrail,
    prevent_target_modification_guardrail,
)
from ds_crew.schemas import CleaningPlan, ColumnCleaningAction, ColumnFeaturePlan, FeatureEngineeringPlan


def _task_output(pydantic_obj):
    # Guardrails only ever touch `.pydantic`; a lightweight stand-in avoids coupling
    # this test to CrewAI's full TaskOutput constructor signature.
    return SimpleNamespace(pydantic=pydantic_obj)


def test_feature_leakage_guardrail_passes_clean_plan(classification_run, run_id):
    plan = FeatureEngineeringPlan(
        run_id=run_id, column_plans=[ColumnFeaturePlan(column="num_a", scaling="standard")]
    )
    ok, result = prevent_target_leakage_guardrail(_task_output(plan))
    assert ok is True
    assert result is plan


def test_feature_leakage_guardrail_rejects_target_as_feature(classification_run, run_id):
    plan = FeatureEngineeringPlan(
        run_id=run_id, column_plans=[ColumnFeaturePlan(column="target", encoding="onehot")]
    )
    ok, error = prevent_target_leakage_guardrail(_task_output(plan))
    assert ok is False
    assert "target" in error


def test_feature_leakage_guardrail_rejects_missing_pydantic():
    ok, error = prevent_target_leakage_guardrail(_task_output(None))
    assert ok is False
    assert "FeatureEngineeringPlan" in error


def test_feature_leakage_guardrail_rejects_unknown_run():
    plan = FeatureEngineeringPlan(run_id="ghost-run", column_plans=[])
    ok, error = prevent_target_leakage_guardrail(_task_output(plan))
    assert ok is False


def test_cleaning_guardrail_passes_clean_plan(classification_run, run_id):
    plan = CleaningPlan(
        run_id=run_id, actions=[ColumnCleaningAction(column="num_a", missing_strategy="mean")]
    )
    ok, result = prevent_target_modification_guardrail(_task_output(plan))
    assert ok is True
    assert result is plan


def test_cleaning_guardrail_rejects_target_action(classification_run, run_id):
    plan = CleaningPlan(
        run_id=run_id, actions=[ColumnCleaningAction(column="target", missing_strategy="mode")]
    )
    ok, error = prevent_target_modification_guardrail(_task_output(plan))
    assert ok is False
    assert "target" in error


def test_cleaning_guardrail_rejects_target_drop(classification_run, run_id):
    plan = CleaningPlan(run_id=run_id, columns_to_drop=["target"])
    ok, error = prevent_target_modification_guardrail(_task_output(plan))
    assert ok is False
    assert "target" in error


def test_cleaning_guardrail_rejects_wrong_type(classification_run, run_id):
    ok, error = prevent_target_modification_guardrail(_task_output("not a plan"))
    assert ok is False


def test_finalize_called_guardrail_rejects_when_never_called(classification_run, run_id):
    guardrail = make_finalize_called_guardrail(run_id)
    ok, error = guardrail(_task_output(None))
    assert ok is False
    assert "finalize_run was not called" in error


def test_finalize_called_guardrail_passes_after_sign_off_recorded(classification_run, run_id):
    classification_run.record("finalize", "sign_off", {"approved": False, "model": "knn"})
    guardrail = make_finalize_called_guardrail(run_id)
    output = _task_output(None)
    ok, result = guardrail(output)
    assert ok is True
    assert result is output


def test_finalize_called_guardrail_ignores_unrelated_history(classification_run, run_id):
    classification_run.record("evaluation", "scored", {"models": ["knn"]})
    guardrail = make_finalize_called_guardrail(run_id)
    ok, error = guardrail(_task_output(None))
    assert ok is False


def test_finalize_called_guardrail_rejects_unknown_run():
    guardrail = make_finalize_called_guardrail("ghost-run")
    ok, error = guardrail(_task_output(None))
    assert ok is False
