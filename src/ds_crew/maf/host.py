"""Run one workflow to completion, plus the operator-facing pieces that don't
belong inside the graph itself: the auto/interactive responders, the final
summary printout, and the pre-flight/run-creation helpers moved here unchanged
from the deleted `ds_crew.foundry.orchestrator`.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from typing import Any

from agent_framework import Workflow

from ds_crew import settings
from ds_crew.foundry.stages import GATED_TOOLS, STAGES_BY_KEY
from ds_crew.maf.executors import VerdictCollector
from ds_crew.maf.state import PipelineState
from ds_crew.maf.transport import Decider, PendingApproval


class PreflightError(RuntimeError):
    """Raised when the environment cannot support a run that would complete."""


def preflight(*, timeout_s: float = 10.0, attempts: int = 3) -> None:
    """Refuse to start a run the environment cannot finish.

    A pipeline that dies partway through leaves a run with some stages'
    one-shot guards already flipped and no way to resume from a fresh run;
    checking first is much cheaper than discovering it later.
    """
    if not settings.AZURE_FOUNDRY_PROJECT_ENDPOINT:
        raise PreflightError(
            "AZURE_FOUNDRY_PROJECT_ENDPOINT is not set. It looks like "
            "https://<account>.services.ai.azure.com/api/projects/<project> and is "
            "distinct from AZURE_FOUNDRY_ENDPOINT."
        )
    if not settings.SERVICE_PUBLIC_URL:
        raise PreflightError(
            "SERVICE_PUBLIC_URL is not set, so Foundry has no address for the tool service."
        )

    health = settings.SERVICE_PUBLIC_URL.rstrip("/") + "/healthz"
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(health, timeout=timeout_s) as resp:
                if resp.status == 200:
                    return
                last = RuntimeError(f"HTTP {resp.status}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < attempts - 1:
            time.sleep(2)
    raise PreflightError(
        f"The tool service is not reachable at {health} after {attempts} attempts "
        f"({last}). Foundry reaches it over the same path, so a run would fail "
        "part-way. Check the tunnel is forwarded and public, and that the service "
        "is running."
    )


def create_run(csv_path: str, target: str, *, task: str = "auto", metric: str | None = None) -> str:
    """Create a run on the tool service and return its id."""
    import pathlib

    payload: dict[str, Any] = {
        "csv_text": pathlib.Path(csv_path).read_text(encoding="utf-8"),
        "target": target,
        "task": task,
    }
    if metric:
        payload["metric"] = metric

    request = urllib.request.Request(
        settings.SERVICE_PUBLIC_URL.rstrip("/") + "/runs",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": settings.SERVICE_API_KEY or ""},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())["run_id"]


def auto_decider(log: Any = print) -> Decider:
    async def decide(req: PendingApproval) -> tuple[bool, str]:
        log(f"    [auto-approved] {req.tool}")
        return True, "AUTO_APPROVE is set."

    return decide


def auto_verdict_collector() -> VerdictCollector:
    async def collect(explanation_text: str) -> str:
        return (
            "No human reviewed this run; it was executed unattended. Record it as NOT approved."
        )

    return collect


def _pretty_arguments(req: PendingApproval) -> str:
    try:
        return json.dumps(json.loads(req.arguments), indent=2)
    except (json.JSONDecodeError, TypeError):
        return req.arguments


def interactive_decider(log: Any = print) -> Decider:
    async def decide(req: PendingApproval) -> tuple[bool, str]:
        log("")
        log("  " + "=" * 68)
        log(f"  HUMAN APPROVAL REQUIRED -- {req.agent} wants to call {req.tool}")
        log("  " + "=" * 68)
        for line in _pretty_arguments(req).splitlines():
            log(f"    {line}")
        log("")
        while True:
            raw = await asyncio.to_thread(
                input, "  Approve? [y]es / [n]o (reason optional after n): "
            )
            answer = raw.strip()
            if not answer:
                continue
            head, _, tail = answer.partition(" ")
            head = head.lower()
            if head in ("y", "yes"):
                return True, tail.strip()
            if head in ("n", "no"):
                return False, tail.strip() or "Rejected by the operator."
            log("  Please answer y or n.")

    return decide


def interactive_verdict_collector(log: Any = print) -> VerdictCollector:
    async def collect(explanation_text: str) -> str:
        log("")
        log("  " + "=" * 68)
        log("  EXPLANATION -- read this before deciding")
        log("  " + "=" * 68)
        for line in explanation_text.splitlines():
            log(f"    {line}")
        log("")
        while True:
            raw = await asyncio.to_thread(
                input,
                "  Approve the recommended model for production? "
                "[y]es / [n]o (reasoning optional after either): ",
            )
            answer = raw.strip()
            if not answer:
                continue
            head, _, tail = answer.partition(" ")
            head = head.lower()
            if head in ("y", "yes"):
                return f"I approve the recommended model. {tail.strip()}".strip()
            if head in ("n", "no"):
                return f"I do NOT approve the recommended model. {tail.strip()}".strip()
            log("  Please answer y or n.")

    return collect


async def drive(
    workflow: Workflow, state: PipelineState | None = None, *, checkpoint_id: str | None = None
) -> PipelineState:
    """Run the workflow to completion and return the final `PipelineState`.

    No `request_info` round trip here: gate decisions and the verdict are
    plain async callbacks baked into the executors when the graph is built
    (see `workflow.build_workflow`), so a run either completes, stops cleanly
    on `GateNotApproved` (both yield the final state), or raises.

    Exactly one of `state` (a fresh run) or `checkpoint_id` (resuming one) is
    expected -- resuming restores the whole `PipelineState` from the
    checkpoint itself, so a `state` passed alongside `checkpoint_id` would
    silently be ignored by `workflow.run`.
    """
    if checkpoint_id is None:
        if state is None:
            raise ValueError("drive() needs either a state (fresh run) or a checkpoint_id (resume).")
        result = await workflow.run(state)
    else:
        result = await workflow.run(checkpoint_id=checkpoint_id)
    outputs = result.get_outputs()
    if not outputs:
        raise RuntimeError("Workflow produced no output -- this should not happen.")
    return outputs[-1]


async def list_checkpoint_ids(checkpoint_storage: Any, *, workflow_name: str) -> list[str]:
    """Checkpoint ids available to `--resume`, most recent last.

    `FileCheckpointStorage.list_checkpoint_ids` returns them in storage order,
    not necessarily chronological, so this isn't itself a "latest" query --
    just the discovery step a human needs before picking one.
    """
    return list(await checkpoint_storage.list_checkpoint_ids(workflow_name=workflow_name))


async def describe_checkpoints(checkpoint_storage: Any, *, workflow_name: str) -> list[dict[str, Any]]:
    """Human-facing detail for `--list-checkpoints`: which run, how far in.

    A `WorkflowCheckpoint`'s own `.state` never carries `PipelineState` -- it
    lives in `.messages`, keyed by whichever executor was next to receive it
    (verified against the installed `agent_framework`, 2026-08-26). Any one
    pending message carries the same `PipelineState`, so the first one found
    is enough to recover `run_id` and how many stages had completed. The
    terminal checkpoint (after `finalize` has already yielded its output) has
    no pending message to peek at, and is reported as such rather than guessed.
    """
    checkpoints = await checkpoint_storage.list_checkpoints(workflow_name=workflow_name)
    rows: list[dict[str, Any]] = []
    for cp in checkpoints:
        run_id = None
        stages_done = None
        for messages in cp.messages.values():
            for msg in messages:
                data = getattr(msg, "data", None)
                if isinstance(data, PipelineState):
                    run_id, stages_done = data.run_id, len(data.results)
                    break
            if run_id is not None:
                break
        rows.append(
            {
                "checkpoint_id": cp.checkpoint_id,
                "timestamp": cp.timestamp,
                "run_id": run_id,
                "stages_done": stages_done,
            }
        )
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def summarize(state: PipelineState, log: Any = print) -> None:
    elapsed = (state.finished_at or time.time()) - state.started_at
    log("")
    log("=" * 78)
    log(f"Run {state.run_id} finished in {elapsed / 60:.1f} min")
    log("=" * 78)
    log(f"{'stage':<16}{'agent':<22}{'in':>8}{'out':>8}  tools")
    for key, result in state.results.items():
        stage = STAGES_BY_KEY[key]
        log(
            f"{key:<16}{stage.agent:<22}{result.input_tokens:>8}{result.output_tokens:>8}  "
            f"{', '.join(result.tool_calls) or '-'}"
        )
    log("")
    log(f"total tokens      : {state.input_tokens} in / {state.output_tokens} out")
    cost = state.cost_usd()
    if cost is None:
        log("estimated cost    : not computed (set LLM_PRICE_PER_1M_INPUT/OUTPUT)")
    else:
        log(f"estimated cost    : ${cost:.4f}")
    if state.transport_retries:
        log(f"transport retries : {state.transport_retries} (tunnel faults, not agent errors)")

    succeeded = {t for r in state.results.values() for t in r.succeeded_tools()}
    gated_called = succeeded & GATED_TOOLS
    log(f"gated tools run   : {', '.join(sorted(gated_called)) or 'none'}")

    expected_gates = {t for s in STAGES_BY_KEY.values() for t in s.expects_tools} & GATED_TOOLS
    if gated_called != expected_gates:
        log(
            f"WARNING: expected {len(expected_gates)} gated tools to run, "
            f"{len(gated_called)} did. Missing: "
            f"{', '.join(sorted(expected_gates - gated_called))}"
        )
