"""FastAPI surface over the deterministic tool layer.

Routes are generated from `registry.TOOL_CLASSES`, so the OpenAPI document this
serves at `/openapi.json` -- the artifact an Azure AI Foundry agent registers
against -- is derived from the same Pydantic argument schemas the in-process
orchestrator validates against. There is no second copy of a tool's contract to
keep in sync.

Deliberately no `from __future__ import annotations` in this module. That import
turns every annotation into a string, and the per-tool request-body type is
attached dynamically (see `_make_tool_endpoint`); FastAPI resolves it by reading
the real class object out of `__annotations__`, which a stringized annotation
would defeat.

## Authorization model

The in-process design gives `run_id` a property worth preserving: it is bound
into each tool's constructor at crew-build time and never exposed as an
LLM-callable argument, so a hallucinating or prompt-injected agent cannot address
another run's data (see state.py). Over HTTP `run_id` is necessarily part of the
request, and a single shared API key would hand every authenticated caller access
to every run -- strictly weaker than what it replaces.

So creating a run mints a **per-run token**, returned once, and every tool call
must present it. An agent handed run A's token cannot reach run B even if it
invents B's id. `SERVICE_API_KEY` gates run *creation*; the run token gates
everything that touches a run's data.
"""

import io
import json
import secrets
import threading
from typing import Any, Literal

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ds_crew import settings
from ds_crew.service.registry import (
    TOOL_CLASSES,
    args_schema_of,
    description_of,
    tool_name_of,
)
from ds_crew.state import get_data_store
from ds_crew.tools.eda_tools import infer_task_type
from ds_crew.tools.model_tools import ALLOWED_METRICS, METRIC_BY_TASK


class RunTokenRegistry:
    """run_id -> run token, for the lifetime of the process.

    Held in memory on purpose, matching `DataStore`: a run's DataFrames only
    exist in this process, so a token that outlived the process would authorize
    access to a run that no longer exists. Both are why the service is
    single-replica for now (see README).
    """

    def __init__(self) -> None:
        self._tokens: dict[str, str] = {}
        self._lock = threading.Lock()

    def mint(self, run_id: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._tokens[run_id] = token
        return token

    def verify(self, run_id: str, presented: str | None) -> bool:
        with self._lock:
            expected = self._tokens.get(run_id)
        if expected is None or not presented:
            return False
        # Constant-time: the token is a bearer credential, and a naive `==`
        # leaks its prefix through timing to a caller who can retry freely.
        return secrets.compare_digest(expected, presented)

    def drop(self, run_id: str) -> None:
        with self._lock:
            self._tokens.pop(run_id, None)


class CreateRunRequest(BaseModel):
    csv_text: str = Field(..., description="The dataset as raw CSV text, including a header row.")
    target: str = Field(..., description="Name of the target column.")
    task: Literal["classification", "regression", "auto"] = "auto"
    test_size: float = Field(default=settings.DEFAULT_TEST_SIZE, gt=0.0, lt=1.0)
    metric: str | None = Field(
        default=None,
        description="Optimization metric. Defaults to a task-appropriate metric.",
    )


class CreateRunResponse(BaseModel):
    run_id: str
    run_token: str = Field(..., description="Present as X-Run-Token on every tool call.")
    task_type: str
    metric_name: str
    n_rows: int
    n_cols: int


class RunStatusResponse(BaseModel):
    run_id: str
    task_type: str | None
    metric_name: str | None
    stages_applied: dict[str, bool]
    history: list[dict[str, Any]]


def _tool_result_to_json(raw: str) -> Any:
    """Tools return a JSON string. Parse it so the HTTP response is real JSON
    rather than a quoted blob, but never fail on a tool that returns plain text.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"result": raw}


def create_app() -> FastAPI:
    app = FastAPI(
        title="DS-Crew tool service",
        version="0.1.0",
        description=(
            "Deterministic, Pydantic-validated tools for the structured-data ML "
            "lifecycle. Create a run, then drive it one tool at a time. Tools "
            "enforce their own ordering and refuse repeat application."
        ),
    )
    tokens = RunTokenRegistry()
    app.state.run_tokens = tokens

    def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
        if not settings.SERVICE_API_KEY:
            raise HTTPException(
                status_code=503,
                detail="SERVICE_API_KEY is not configured; the service refuses to run open.",
            )
        if not secrets.compare_digest(x_api_key, settings.SERVICE_API_KEY):
            raise HTTPException(status_code=401, detail="Invalid API key.")

    def require_run_token(run_id: str, x_run_token: str = Header(..., alias="X-Run-Token")) -> str:
        if not tokens.verify(run_id, x_run_token):
            # 404 rather than 403 on purpose: a caller holding no valid token for
            # this run should not be able to use the status code to learn whether
            # the run exists.
            raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.")
        return run_id

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "tools": [tool_name_of(c) for c in TOOL_CLASSES]}

    @app.post(
        "/runs",
        response_model=CreateRunResponse,
        tags=["runs"],
        dependencies=[Depends(require_api_key)],
    )
    def create_run(body: CreateRunRequest) -> CreateRunResponse:
        try:
            df = pd.read_csv(io.StringIO(body.csv_text))
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse csv_text: {exc}") from exc

        if body.target not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"target '{body.target}' not found in columns: {list(df.columns)}",
            )

        task_type = body.task if body.task != "auto" else infer_task_type(df[body.target])

        if body.metric and body.metric not in ALLOWED_METRICS[task_type]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"metric '{body.metric}' not allowed for task type '{task_type}': "
                    f"{sorted(ALLOWED_METRICS[task_type])}"
                ),
            )

        run_id = secrets.token_hex(6)
        state = get_data_store().create_run(run_id, df, target=body.target)
        state.task_type = task_type
        state.test_size = body.test_size
        state.metric_name = body.metric or METRIC_BY_TASK[task_type]

        return CreateRunResponse(
            run_id=run_id,
            run_token=tokens.mint(run_id),
            task_type=task_type,
            metric_name=state.metric_name,
            n_rows=len(df),
            n_cols=len(df.columns),
        )

    @app.get("/runs/{run_id}", response_model=RunStatusResponse, tags=["runs"])
    def get_run(run_id: str = Depends(require_run_token)) -> RunStatusResponse:
        state = get_data_store().get(run_id)
        return RunStatusResponse(
            run_id=run_id,
            task_type=state.task_type,
            metric_name=state.metric_name,
            stages_applied={
                "cleaning": state.cleaning_applied,
                "features": state.features_applied,
                "ensemble": state.ensemble_applied,
                "evaluation": state.evaluation_applied,
                "explanation": state.explanation_applied,
                "finalize": state.finalize_applied,
            },
            history=state.history,
        )

    @app.delete("/runs/{run_id}", tags=["runs"], status_code=204)
    def delete_run(run_id: str = Depends(require_run_token)) -> None:
        get_data_store().drop(run_id)
        tokens.drop(run_id)

    for tool_cls in TOOL_CLASSES:
        app.add_api_route(
            f"/runs/{{run_id}}/tools/{tool_name_of(tool_cls)}",
            _make_tool_endpoint(tool_cls, require_run_token),
            methods=["POST"],
            tags=["tools"],
            name=tool_name_of(tool_cls),
            summary=tool_name_of(tool_cls),
            description=description_of(tool_cls),
        )

    return app


def _make_tool_endpoint(tool_cls: type, require_run_token: Any) -> Any:
    """Build the POST handler for one tool, with its args_schema as the body model.

    The body type is attached via `__annotations__` rather than written literally,
    because it differs per tool and FastAPI needs the concrete class to generate
    that route's OpenAPI schema. See the module docstring on why this file must
    not use postponed annotation evaluation.
    """
    schema = args_schema_of(tool_cls)

    def endpoint(payload, run_id: str = Depends(require_run_token)) -> Any:
        body = payload.model_dump()

        # Several argument schemas are the pipeline's plan/report models, which
        # carry their own run_id so guardrails can find their way back into the
        # DataStore. The path is authoritative -- it is what the run token was
        # checked against. A mismatch is rejected rather than quietly rewritten,
        # matching how the tools themselves treat a wrong run_id: a caller that
        # disagrees with itself about which run it is driving has a bug worth
        # surfacing, and silently correcting it would hide exactly the confusion
        # the *_applied flags exist to catch.
        declared = body.get("run_id")
        if declared and declared != run_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"run_id in body ('{declared}') does not match the path ('{run_id}')."
                ),
            )
        if "run_id" in body:
            body["run_id"] = run_id

        try:
            get_data_store().get(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        # `_run` rather than `run`: it is the deterministic body the in-process
        # orchestrator ultimately reaches, without CrewAI's LLM-facing result
        # coercion in between. Tools signal refusal by returning an {"error": ...}
        # payload, which is passed through with a 200 exactly as an agent would
        # see it in-process -- an out-of-order call is a tool-level decision, not
        # an HTTP-level failure.
        raw = tool_cls(run_id=run_id)._run(**body)
        return _tool_result_to_json(raw)

    endpoint.__annotations__["payload"] = schema
    endpoint.__name__ = f"{tool_name_of(tool_cls)}_endpoint"
    return endpoint


app = create_app()
