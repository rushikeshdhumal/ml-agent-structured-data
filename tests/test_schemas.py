from __future__ import annotations

import pytest
from pydantic import ValidationError

from ds_crew.schemas import (
    DTYPE_CASTS,
    ENCODING_STRATEGIES,
    MISSING_STRATEGIES,
    OUTLIER_STRATEGIES,
    SCALING_STRATEGIES,
    SELECTION_METHODS,
    ColumnCleaningAction,
    ColumnFeaturePlan,
    EdaReport,
    EvaluationReport,
    FeatureEngineeringPlan,
    _canonical,
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


# ---------------------------------------------------------------------------
# Enum spelling normalization
# ---------------------------------------------------------------------------


def test_column_feature_plan_accepts_one_hot_spelling():
    # Observed live: a Foundry agent on gpt-5-mini, given the fully inlined
    # schema including the enum, still sent "one_hot". It is the near-universal
    # spelling (sklearn's OneHotEncoder), so the model was following a strong
    # prior rather than ignoring the schema.
    plan = ColumnFeaturePlan(column="cat_a", encoding="one_hot")
    assert plan.encoding == "onehot"


@pytest.mark.parametrize(
    ("field", "written", "canonical"),
    [
        ("encoding", "one-hot", "onehot"),
        ("encoding", "OneHot", "onehot"),
        ("encoding", "target mean", "target_mean"),
        ("scaling", "min_max", "minmax"),
        ("scaling", "MinMax", "minmax"),
        ("scaling", "Standard", "standard"),
    ],
)
def test_feature_enum_separator_and_case_variants(field, written, canonical):
    plan = ColumnFeaturePlan(column="c", **{field: written})
    assert getattr(plan, field) == canonical


@pytest.mark.parametrize(
    ("field", "written", "canonical"),
    [
        ("missing_strategy", "drop-rows", "drop_rows"),
        ("missing_strategy", "Median", "median"),
        ("outlier_strategy", "iqr clip", "iqr_clip"),
        ("outlier_strategy", "z_score_clip", "zscore_clip"),
        ("dtype_cast", "Float", "float"),
    ],
)
def test_cleaning_enum_separator_and_case_variants(field, written, canonical):
    action = ColumnCleaningAction(column="c", **{field: written})
    assert getattr(action, field) == canonical


def test_normalization_does_not_invent_synonyms():
    """Spelling only. A guess about intent would be a different feature."""
    with pytest.raises(ValidationError):
        ColumnCleaningAction(column="c", missing_strategy="forward_fill")
    with pytest.raises(ValidationError):
        ColumnFeaturePlan(column="c", encoding="dummy")
    with pytest.raises(ValidationError):
        ColumnFeaturePlan(column="c", scaling="zscore")


def test_enum_aliases_cannot_collide():
    """The docstring on `enum_alias` promises this; assert it rather than trust it.

    Normalization is only safe because no two options in one enum are identical
    once separators and case are stripped. Adding e.g. "one_hot" alongside
    "onehot" would silently make one unreachable.
    """
    enums = {
        "MISSING_STRATEGIES": MISSING_STRATEGIES,
        "OUTLIER_STRATEGIES": OUTLIER_STRATEGIES,
        "DTYPE_CASTS": DTYPE_CASTS,
        "ENCODING_STRATEGIES": ENCODING_STRATEGIES,
        "SCALING_STRATEGIES": SCALING_STRATEGIES,
        "SELECTION_METHODS": SELECTION_METHODS,
    }
    for name, values in enums.items():
        canonical = [_canonical(v) for v in values]
        assert len(set(canonical)) == len(values), f"{name} collides: {canonical}"
