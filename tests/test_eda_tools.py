from __future__ import annotations

import json

import pandas as pd
import pytest

from ds_crew.schemas import EdaReport
from ds_crew.tools.eda_tools import EdaSummaryTool, build_eda_report, infer_task_type


def test_infer_task_type_object_dtype_is_classification():
    y = pd.Series(["a", "b", "a", "b"])
    assert infer_task_type(y) == "classification"


def test_infer_task_type_low_cardinality_int_is_classification():
    y = pd.Series([0, 1, 0, 1, 1, 0])
    assert infer_task_type(y, max_classes=5) == "classification"


def test_infer_task_type_high_cardinality_numeric_is_regression():
    y = pd.Series(range(100))
    assert infer_task_type(y, max_classes=5) == "regression"


def test_build_eda_report_classification(classification_df):
    report = build_eda_report(classification_df, "target", "classification")
    assert report.n_rows == len(classification_df)
    assert report.n_cols == len(classification_df.columns)
    assert report.class_balance is not None
    assert set(report.class_balance) == {"yes", "no"}
    names = {c.name for c in report.columns}
    assert "constant_col" in names
    constant_profile = next(c for c in report.columns if c.name == "constant_col")
    assert constant_profile.is_constant
    id_profile = next(c for c in report.columns if c.name == "cat_b_high_card")
    assert id_profile.is_id_like
    assert any("cat_b_high_card" in flag for flag in report.leakage_flags)


def test_build_eda_report_regression_has_correlations(regression_df):
    report = build_eda_report(regression_df, "target", "regression")
    assert "num_a" in report.correlations_with_target
    assert report.class_balance is None


def test_build_eda_report_truncates_wide_datasets(regression_df):
    report = build_eda_report(regression_df, "target", "regression", detailed_column_limit=2)
    assert report.truncated is True
    assert len(report.columns) == 2
    assert "target" in {c.name for c in report.columns}


def test_near_duplicate_columns_detected():
    df = pd.DataFrame({"a": range(50), "b": range(50), "target": [0, 1] * 25})
    report = build_eda_report(df, "target", "classification")
    assert ("a", "b") in report.near_duplicate_column_pairs


def test_eda_summary_tool_returns_valid_report_json(classification_run, run_id):
    tool = EdaSummaryTool(run_id=run_id)
    result = tool._run(include_correlations=True)
    report = EdaReport.model_validate(json.loads(result))
    assert report.run_id == run_id
    assert report.n_rows == len(classification_run.df)


def test_eda_summary_tool_unknown_run_raises():
    tool = EdaSummaryTool(run_id="nope")
    with pytest.raises(KeyError):
        tool._run()


def test_eda_summary_tool_records_history(classification_run, run_id):
    tool = EdaSummaryTool(run_id=run_id)
    tool._run()
    assert classification_run.history[-1]["stage"] == "eda"
