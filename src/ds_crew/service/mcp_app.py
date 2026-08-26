"""MCP surface over the same tool registry the HTTP API exposes.

Azure AI Foundry's gpt-5-family deployments advertise `agentsV2` and offer the
**mcp** custom tool but not `openapi`, so this is the only way a Foundry agent
can reach these tools. It is also the better surface regardless: the `mcp` tool
type is the only one carrying `require_approval`, which is what restores
DS-Crew's human approval gates.

Tools come from `registry.TOOL_CLASSES`, the same source `app.py` builds its
REST routes from. One tool contract, two protocol surfaces, no second copy to
drift.
"""

from __future__ import annotations

import inspect
import secrets
from typing import Any
from urllib.parse import urlparse

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from ds_crew import settings
from ds_crew.service.registry import (
    TOOL_CLASSES,
    args_schema_of,
    description_of,
    tool_name_of,
)
from ds_crew.state import get_data_store

MCP_MOUNT_PATH = "/mcp"

# Always permitted so local development and the test suite work without
# SERVICE_PUBLIC_URL being set. Both bare and port-suffixed forms appear
# depending on the client.
_LOCAL_HOSTS = ["localhost", "localhost:8000", "127.0.0.1", "127.0.0.1:8000", "testserver"]


def _allowed_hosts() -> list[str]:
    """Hosts the MCP transport will accept, derived from `SERVICE_PUBLIC_URL`.

    Returning a concrete allowlist keeps DNS-rebinding protection switched on.
    Disabling it would be the easy fix and a worse one: this endpoint can train
    models and mutate datasets.
    """
    hosts = list(_LOCAL_HOSTS)
    if settings.SERVICE_PUBLIC_URL:
        parsed = urlparse(settings.SERVICE_PUBLIC_URL)
        if parsed.hostname:
            hosts.append(parsed.hostname)
            if parsed.port:
                hosts.append(f"{parsed.hostname}:{parsed.port}")
            # A tunnel or ingress terminates TLS on 443 and forwards without a
            # port, but some clients still send the explicit one.
            hosts.append(f"{parsed.hostname}:443")
    return hosts


def _make_tool_callable(tool_cls: type) -> Any:
    """Build the async callable FastMCP will expose for one tool.

    **Flat signature, built dynamically.** FastMCP derives a tool's inputSchema
    from the function signature, and a single Pydantic parameter would nest
    every field under one property with a `$defs` indirection. Constructing
    explicit keyword parameters instead yields a flat schema, which is markedly
    easier for a model to fill in correctly. `$defs` still appears where a model
    genuinely nests (cleaning actions, column plans), which is correct.

    **`run_id` is always a parameter.** Four argument schemas
    (CleaningPlan, FeatureEngineeringPlan, MetricChoice, FinalSignOff) already
    carry it because guardrails need it; the rest get it prepended. Foundry's
    mcp tool sends only static `headers`, so a per-run credential is
    unrepresentable and the run id has to travel as an argument.

    **Async, with the real work off the event loop.** These tools train models,
    run Optuna searches and compute SHAP values; `tune_model_hyperparameters`
    alone is bounded by `MAX_HPO_TIMEOUT_S` at 30 minutes. Running that inline
    would block the whole ASGI server, so `/healthz` would stop answering and no
    other tool could be called for the duration.
    """
    schema = args_schema_of(tool_cls)
    fields = schema.model_fields
    schema_has_run_id = "run_id" in fields

    params: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}

    if not schema_has_run_id:
        params.append(inspect.Parameter("run_id", inspect.Parameter.KEYWORD_ONLY, annotation=str))
        annotations["run_id"] = str

    for name, field in fields.items():
        default = (
            inspect.Parameter.empty
            if field.is_required()
            else field.get_default(call_default_factory=True)
        )
        params.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=field.annotation,
                default=default,
            )
        )
        annotations[name] = field.annotation

    async def tool_callable(**kwargs: Any) -> str:
        run_id = kwargs["run_id"] if schema_has_run_id else kwargs.pop("run_id")
        if not run_id:
            return '{"error": "run_id is required."}'
        try:
            get_data_store().get(run_id)
        except KeyError as exc:
            # Returned rather than raised: the tools signal refusal in-band with
            # an {"error": ...} payload and the agent is expected to read and
            # correct course. A protocol-level error would just look like an
            # outage to it.
            return f'{{"error": {str(exc)!r}}}'
        return await anyio.to_thread.run_sync(
            lambda: tool_cls(run_id=run_id)._run(**kwargs)
        )

    tool_callable.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    tool_callable.__annotations__ = {**annotations, "return": str}
    tool_callable.__name__ = tool_name_of(tool_cls)
    return tool_callable


def build_mcp_server() -> FastMCP:
    """A FastMCP server exposing every tool in the registry.

    `stateless_http=True` because each tool call is already self-contained: the
    run lives in the process-wide `DataStore` keyed by `run_id`, not in MCP
    session state. A hosted agent reconnecting mid-pipeline therefore loses
    nothing, and there is no per-session state to lose.

    The MCP SDK enables **DNS-rebinding protection** by default with an empty
    host allowlist, which rejects any Host header it does not recognise with
    HTTP 421. Left alone that rejects Foundry outright, since the request
    arrives with the tunnel's (or Container App's) hostname. Rather than switch
    the protection off, the allowlist is derived from `SERVICE_PUBLIC_URL`,
    which has to be set for this deployment anyway, so it stays correct as the
    tunnel URL rotates.
    """
    mcp = FastMCP(
        "ds-crew-tools",
        transport_security=TransportSecuritySettings(allowed_hosts=_allowed_hosts()),
        instructions=(
            "Deterministic, Pydantic-validated tools for the structured-data ML "
            "lifecycle. Every tool operates on an existing run identified by "
            "run_id, which the operator creates out of band. Tools enforce their "
            "own ordering and refuse repeat application; an {\"error\": ...} "
            "result is a considered refusal to act on, not a transport failure."
        ),
        stateless_http=True,
        streamable_http_path=MCP_MOUNT_PATH,
    )
    for tool_cls in TOOL_CLASSES:
        mcp.add_tool(
            _make_tool_callable(tool_cls),
            name=tool_name_of(tool_cls),
            description=description_of(tool_cls),
        )
    return mcp


class RequireApiKeyASGI:
    """Gate the mounted MCP app on `X-API-Key`.

    Foundry's mcp tool sends a fixed `headers` map, so a static key is the only
    credential it can present. Implemented as ASGI middleware rather than a
    FastAPI dependency because the MCP app is a mounted Starlette application
    and never passes through FastAPI's dependency machinery.

    Same posture as the REST surface: refuse to serve at all when
    SERVICE_API_KEY is unset, rather than defaulting to an open endpoint that
    can mutate datasets and train models.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def _reject(self, send: Any, status: int, message: str) -> None:
        body = f'{{"error": "{message}"}}'.encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # The wrapped app is mounted at "/" so it receives every path FastAPI
        # did not match. Guard only the MCP endpoint and let anything else fall
        # through to a plain 404: answering a mistyped URL with 401 is
        # confusing, and implies a protected resource exists where none does.
        if not scope.get("path", "").startswith(MCP_MOUNT_PATH):
            await self.app(scope, receive, send)
            return
        if not settings.SERVICE_API_KEY:
            await self._reject(send, 503, "SERVICE_API_KEY is not configured.")
            return
        presented = ""
        for key, value in scope.get("headers", []):
            if key == b"x-api-key":
                presented = value.decode("latin-1")
                break
        if not presented or not secrets.compare_digest(presented, settings.SERVICE_API_KEY):
            await self._reject(send, 401, "Invalid or missing API key.")
            return
        await self.app(scope, receive, send)
