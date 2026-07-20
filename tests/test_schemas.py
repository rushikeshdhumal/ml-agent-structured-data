from __future__ import annotations

from ds_crew.schemas import (
    ColumnCleaningAction,
    EdaReport,
    EvaluationReport,
    FeatureEngineeringPlan,
)


def test_feature_engineering_plan_coerces_string_null_top_k():
    # Observed live: NVIDIA's z-ai/glm-5.2 emitted the literal string "null"
    # instead of JSON null for an Optional int field via structured tool-call args.
    plan = FeatureEngineeringPlan(run_id="r", top_k="null")
    assert plan.top_k is None


def test_feature_engineering_plan_accepts_real_int_top_k():
    plan = FeatureEngineeringPlan(run_id="r", top_k=5)
    assert plan.top_k == 5


def test_column_cleaning_action_coerces_string_null_fields():
    action = ColumnCleaningAction(
        column="x",
        missing_strategy="null",
        constant_fill_value="None",
        dtype_cast="null",
    )
    assert action.missing_strategy is None
    assert action.constant_fill_value is None
    assert action.dtype_cast is None


def test_eda_report_coerces_string_null_class_balance():
    report = EdaReport(
        run_id="r",
        n_rows=10,
        n_cols=2,
        target="y",
        task_type="regression",
        columns=[],
        class_balance="null",
    )
    assert report.class_balance is None


def test_evaluation_report_coerces_string_null_optional_fields():
    report = EvaluationReport(
        model_name="m",
        metrics={"r2": 0.9},
        confusion_matrix="null",
        feature_importances="null",
    )
    assert report.confusion_matrix is None
    assert report.feature_importances is None
