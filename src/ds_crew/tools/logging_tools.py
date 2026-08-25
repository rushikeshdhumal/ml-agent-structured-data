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
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
from crewai.tools import BaseTool
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

from ds_crew import settings
from ds_crew.schemas import FinalSignOff
from ds_crew.state import get_data_store

# Model-library-specific MLflow flavors preserve native serialization
# robustness (CatBoost's own docs specifically recommend this over pickling)
# and correct metadata. Not used for xgboost: mlflow.xgboost.save_model()
# raises on model_tools.py's _XGBClassifierWithLabelEncoding (it isn't a
# plain xgboost.sklearn estimator), and even for a plain XGBClassifier its
# native format only captures the underlying booster, silently dropping any
# extra Python-level state -- the generic sklearn/cloudpickle flavor below
# preserves the whole object instead, which is what a custom subclass needs.
_FLAVOR_BY_MODULE_PREFIX = {
    "lightgbm": mlflow.lightgbm,
    "catboost": mlflow.catboost,
}


def _flavor_for(model: Any):
    return _FLAVOR_BY_MODULE_PREFIX.get(type(model).__module__.split(".")[0], mlflow.sklearn)


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


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimated USD for one run's LLM traffic, or None when rates aren't set.

    Returning None (rather than 0.0) when either rate is unconfigured is the
    point: it records "cost unknown / not priced" instead of asserting the run
    was free, which would be a real claim and usually a wrong one. Callers must
    omit the metric entirely on None.
    """
    price_in = settings.LLM_PRICE_PER_1M_INPUT
    price_out = settings.LLM_PRICE_PER_1M_OUTPUT
    if price_in is None or price_out is None:
        return None
    return round(
        (prompt_tokens / 1_000_000) * price_in + (completion_tokens / 1_000_000) * price_out,
        6,
    )


def log_llm_usage(run_id: str | None, crew: Any, wall_clock_s: float) -> None:
    """Record what a run actually consumed: tokens, request count, wall clock,
    and (when priced) an estimated cost.

    Reads `crew.calculate_usage_metrics()` rather than `CrewOutput.token_usage`
    because this is called from a `finally` and must work on the failure path
    too -- a crashed run is precisely the one whose spend you want recorded, and
    on that path there is no CrewOutput at all.

    Never raises. Being called from a `finally` means any exception escaping
    here would *replace* the real exception propagating out of kickoff, turning
    a diagnosable pipeline failure into a confusing logging failure. Losing a
    metric is strictly better than losing a stack trace.
    """
    metrics: dict[str, float] = {"wall_clock_s": round(wall_clock_s, 1)}
    try:
        usage = crew.calculate_usage_metrics()
        for field in (
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "cached_prompt_tokens",
            "successful_requests",
        ):
            value = getattr(usage, field, None)
            if value is not None:
                metrics[field] = float(value)
        cost = estimate_cost_usd(
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )
        if cost is not None:
            metrics["estimated_cost_usd"] = cost
        log_metrics(run_id, metrics)
    except Exception as exc:  # noqa: BLE001 -- see docstring: must not mask kickoff's error
        print(f"Warning: could not record LLM usage metrics ({type(exc).__name__}: {exc})")


def log_model_artifact(run_id: str | None, model: Any, artifact_path: str = "model") -> None:
    """Saves `model` locally via the flavor matching its library, then uploads
    that directory as an artifact against run_id directly through
    MlflowClient -- not the fluent mlflow.start_run()/log_model() API, which
    has no run_id parameter and only knows how to log into whatever run is
    "active" in the CURRENT thread. Re-entering the run via
    `with mlflow.start_run(run_id=run_id):` would work for logging, but
    exiting that block terminates the run (sets it FINISHED) immediately --
    even though the crew's own outer run context in main.py is still open.
    Going through MlflowClient sidesteps run-lifecycle state entirely.
    """
    if not run_id:
        return
    flavor = _flavor_for(model)
    tmp_dir = tempfile.mkdtemp(prefix="ds_crew_model_")
    try:
        model_path = str(Path(tmp_dir) / "model")
        flavor.save_model(model, model_path)
        MlflowClient().log_artifacts(run_id, model_path, artifact_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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

        if state.finalize_applied:
            return json.dumps(
                {
                    "error": "finalize_run has already been called for this run. Do not call "
                    "it again."
                }
            )

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

        state.finalize_applied = True
        state.record(
            "finalize", "sign_off", {"approved": signoff.approved, "model": signoff.selected_model}
        )
        return json.dumps(
            {
                "status": "approved" if signoff.approved else "rejected",
                "selected_model": signoff.selected_model,
            }
        )
