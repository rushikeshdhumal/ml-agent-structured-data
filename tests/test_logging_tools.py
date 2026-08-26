from __future__ import annotations

import json

import mlflow
import pytest

from ds_crew.schemas import CleaningPlan, ColumnCleaningAction, EvaluationReport
from ds_crew.tools.logging_tools import (
    FinalizeRunTool,
    log_json_artifact,
    log_metrics,
    log_params,
    log_plan_and_feedback,
    log_tags,
)


@pytest.fixture
def mlflow_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-experiment")
    with mlflow.start_run() as run:
        yield run
    mlflow.set_tracking_uri("")


def test_logging_helpers_are_noop_without_run_id():
    # No run_id -- these must not raise and must not create a run as a side effect.
    log_params(None, {"x": 1})
    log_metrics(None, {"y": 1.0})
    log_tags(None, {"z": "tag"})
    log_json_artifact(None, {"a": 1}, "test.json")


def test_log_params_and_metrics_land_via_explicit_run_id(mlflow_run):
    # Deliberately exit the ambient `with mlflow.start_run()` context that created
    # this run before logging, to prove these calls don't depend on it being
    # "active" in the calling thread at all -- see logging_tools.py's docstring
    # on why MlflowClient is used over the fluent API in the first place.
    mlflow.end_run()
    run_id = mlflow_run.info.run_id

    log_params(run_id, {"cv_folds": 5})
    log_metrics(run_id, {"accuracy": 0.9})
    log_tags(run_id, {"my_tag": "completed"})

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    assert run.data.params["cv_folds"] == "5"
    assert run.data.metrics["accuracy"] == 0.9
    assert run.data.tags["my_tag"] == "completed"


def test_log_plan_and_feedback_writes_artifacts(mlflow_run):
    mlflow.end_run()
    run_id = mlflow_run.info.run_id
    plan = CleaningPlan(
        run_id="r", actions=[ColumnCleaningAction(column="x", missing_strategy="mean")]
    )
    log_plan_and_feedback(run_id, plan, stage="cleaning", human_feedback="looks good")
    client = mlflow.tracking.MlflowClient()
    artifacts = {a.path for a in client.list_artifacts(run_id, "cleaning")}
    assert "cleaning/plan.json" in artifacts
    assert "cleaning/human_feedback.txt" in artifacts


def test_finalize_run_tool_requires_evaluation_report(classification_run, run_id):
    tool = FinalizeRunTool(run_id=run_id)
    result = json.loads(tool._run(selected_model="random_forest", approved=True))
    assert "error" in result


def test_finalize_run_tool_approved_updates_state(classification_run, run_id, mlflow_run):
    mlflow.end_run()
    classification_run.mlflow_run_id = mlflow_run.info.run_id
    classification_run.evaluation_reports["random_forest"] = EvaluationReport(
        model_name="random_forest", metrics={"accuracy": 0.9}
    )

    class DummyModel:
        pass

    classification_run.fitted_models["random_forest"] = DummyModel()
    tool = FinalizeRunTool(run_id=run_id)
    result = json.loads(
        tool._run(selected_model="random_forest", approved=True, human_feedback="ship it")
    )
    assert result["status"] == "approved"
    assert classification_run.history[-1]["stage"] == "finalize"
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(mlflow_run.info.run_id)
    assert run.data.tags["model_status"] == "approved"


def test_finalize_run_tool_rejected_logs_no_model(classification_run, run_id, mlflow_run):
    mlflow.end_run()
    classification_run.mlflow_run_id = mlflow_run.info.run_id
    classification_run.evaluation_reports["random_forest"] = EvaluationReport(
        model_name="random_forest", metrics={"accuracy": 0.9}
    )
    tool = FinalizeRunTool(run_id=run_id)
    result = json.loads(
        tool._run(selected_model="random_forest", approved=False, human_feedback="not good enough")
    )
    assert result["status"] == "rejected"
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(mlflow_run.info.run_id)
    assert run.data.tags["model_status"] == "rejected"
    artifacts = {a.path for a in client.list_artifacts(mlflow_run.info.run_id)}
    assert "model" not in artifacts


def test_finalize_run_tool_still_works_without_mlflow_run_id(classification_run, run_id):
    # state.mlflow_run_id defaults to None (e.g. a tool called directly, no MLflow
    # run wired up) -- finalize must still succeed and just skip all logging.
    classification_run.evaluation_reports["random_forest"] = EvaluationReport(
        model_name="random_forest", metrics={"accuracy": 0.9}
    )
    tool = FinalizeRunTool(run_id=run_id)
    result = json.loads(tool._run(selected_model="random_forest", approved=False))
    assert result["status"] == "rejected"
