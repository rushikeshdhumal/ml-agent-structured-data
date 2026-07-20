"""MLflow metadata logging.

Mostly plain functions (not agent-facing tools) called directly from inside
each mutating tool's `_run()` and from `main.py`'s run lifecycle -- logging
has no reasoning content, so it is deterministic code wired to always fire
(including on failure), never an agent's responsibility to remember.

Every helper takes the target run's `run_id` explicitly (threaded through
from `RunState.mlflow_run_id`, set once in `main.py`) and logs via
`MlflowClient` rather than the ambient `mlflow.log_*`/`mlflow.active_run()`
fluent API. That matters: CrewAI executes tool `_run()` calls from a worker
thread, not the thread that opened the run in `main.py`, and
`mlflow.active_run()` is thread-local -- it returns None from inside a tool
even while the run is genuinely active, silently swallowing every log call.
`MlflowClient` methods take run_id directly and don't depend on which
thread they're called from. When `run_id` is None (e.g. a tool's `_run()`
called directly in a unit test, with no MLflow run wired up), every helper
is a no-op.

`FinalizeRunTool` is the one agent-facing exception in this module:
finalization needs an agent to communicate a human's approve/reject
decision through a tool call so it can be validated and recorded like any
other action. Its outcome is tagged `model_status` (approved/rejected) --
deliberately not `status`, which `main.py` already uses for the crew run's
own completed/failed outcome; sharing one key would let whichever write
happens last silently clobber the other.
"""

from __future__ import annotations

import json
from typing import Any

import mlflow
import mlflow.sklearn
from crewai.tools import BaseTool
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

from ds_crew.schemas import FinalSignOff
from ds_crew.state import get_data_store


def log_params(run_id: str | None, params: dict[str, Any]) -> None:
    if not run_id or not params:
        return
    client = MlflowClient()
    for key, value in params.items():
        client.log_param(run_id, key, value)


def log_metrics(run_id: str | None, metrics: dict[str, float]) -> None:
    if not run_id or not metrics:
        return
    client = MlflowClient()
    for key, value in metrics.items():
        client.log_metric(run_id, key, value)


def log_tags(run_id: str | None, tags: dict[str, Any]) -> None:
    if not run_id or not tags:
        return
    client = MlflowClient()
    for key, value in tags.items():
        client.set_tag(run_id, key, value)


def log_json_artifact(run_id: str | None, obj: BaseModel | dict, artifact_file: str) -> None:
    if not run_id:
        return
    payload = obj.model_dump() if isinstance(obj, BaseModel) else obj
    MlflowClient().log_dict(run_id, payload, artifact_file)


def log_text_artifact(run_id: str | None, text: str, artifact_file: str) -> None:
    if not run_id or not text:
        return
    MlflowClient().log_text(run_id, text, artifact_file)


def log_plan_and_feedback(
    run_id: str | None, plan: BaseModel, stage: str, human_feedback: str = ""
) -> None:
    """Logs a human-reviewed plan (and any raw feedback text) as JSON/text artifacts."""
    log_json_artifact(run_id, plan, f"{stage}/plan.json")
    log_text_artifact(run_id, human_feedback, f"{stage}/human_feedback.txt")


def log_stage_metrics(run_id: str | None, stage: str, metrics: dict[str, float]) -> None:
    log_metrics(run_id, {f"{stage}_{k}": v for k, v in metrics.items()})


def log_model_artifact(run_id: str | None, model: Any, artifact_path: str = "model") -> None:
    if not run_id:
        return
    # mlflow.sklearn.log_model has no run_id parameter -- it only knows how to log
    # into whatever run is "active" in the CURRENT thread. Re-entering the run by
    # id makes it active here regardless of which thread this is.
    with mlflow.start_run(run_id=run_id):
        mlflow.sklearn.log_model(model, artifact_path)


class FinalizeRunTool(BaseTool):
    name: str = "finalize_run"
    description: str = (
        "Records the human's final sign-off decision for the run. If approved, logs the "
        "selected model as an MLflow artifact and tags model_status=approved. If "
        "rejected, tags model_status=rejected and logs no model artifact. Call exactly "
        "once, after evaluation."
    )
    args_schema: type[BaseModel] = FinalSignOff
    run_id: str = ""

    def _run(self, **kwargs) -> str:
        kwargs.setdefault("run_id", self.run_id)
        signoff = FinalSignOff(**kwargs)
        state = get_data_store().get(self.run_id)

        if signoff.selected_model not in state.evaluation_reports:
            return json.dumps(
                {
                    "error": (
                        f"'{signoff.selected_model}' has no evaluation report; "
                        "evaluate it first."
                    )
                }
            )

        mlflow_run_id = state.mlflow_run_id
        log_text_artifact(mlflow_run_id, signoff.human_feedback, "finalize/human_feedback.txt")

        if signoff.approved:
            model = state.fitted_models[signoff.selected_model]
            log_model_artifact(mlflow_run_id, model, artifact_path="model")
            log_json_artifact(
                mlflow_run_id,
                state.evaluation_reports[signoff.selected_model],
                "final_evaluation_report.json",
            )
            log_tags(
                mlflow_run_id,
                {"model_status": "approved", "selected_model": signoff.selected_model},
            )
        else:
            log_tags(
                mlflow_run_id,
                {"model_status": "rejected", "selected_model": signoff.selected_model},
            )

        state.record(
            "finalize", "sign_off", {"approved": signoff.approved, "model": signoff.selected_model}
        )
        return json.dumps(
            {
                "status": "approved" if signoff.approved else "rejected",
                "selected_model": signoff.selected_model,
            }
        )
