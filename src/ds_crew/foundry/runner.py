"""Transport-fault classification, in-band refusal detection, and the result
dataclasses `ds_crew.maf` builds on -- what's left of the hand-written
Foundry orchestrator after `ds_crew.maf` (Microsoft Agent Framework) replaced
its actual conversation-driving loop (the old `AgentRunner`, deleted here
once nothing referenced it any more; see `ds_crew.maf.transport_foundry` for
the transport that replaced it).

Foundry enumerates the MCP tool list at the start of **every** invocation, so
a single dropped `initialize` fails the whole call before the model does any
work, surfacing as `tool_user_error: Initialization timed out`. That is a
transport fault, not an agent fault, and has to be retried rather than
reported as a failed stage -- `is_transport_error` below is what
`ds_crew.maf.transport.with_transport_retries` classifies it with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

# Substrings that mark a failure as transport rather than agent behaviour. Kept
# as substrings because Foundry reports several distinct relay faults through
# one `tool_user_error` code.
_TRANSPORT_MARKERS = (
    "initialization timed out",
    "tool_user_error",
    "enumerating tools from remote server",
    "did not complete the request within the configured timeout",
    "502",
    "503",
    "504",
    "gateway",
)


def is_transport_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(m in text for m in _TRANSPORT_MARKERS)


def _is_refusal(item: Any) -> bool:
    """Did this tool call come back with an in-band `{"error": ...}` payload?

    DS-Crew's tools refuse out-of-order and repeat invocations by returning an
    error object with HTTP 200, so the transport reports success either way.
    Treating that as a completed step is what let a run march through nine
    stages and apply nothing.
    """
    if getattr(item, "error", None):
        return True
    output = getattr(item, "output", None)
    if not output:
        return False
    if not isinstance(output, str):
        output = str(output)
    try:
        parsed = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return '"error"' in output
    return isinstance(parsed, dict) and "error" in parsed


@dataclass(frozen=True)
class ToolEvent:
    """One text/tool-call/tool-result item, in the order it actually occurred.

    Exists because `tool_calls`/`tool_results` below lose two things
    `ds_crew.maf.azure_evaluation` needs to reconstruct a real conversation
    for Azure AI Evaluation SDK evaluators: relative order (interleaving text
    with calls/results) and per-call arguments. `call_id` links a
    `tool_call` event to its `tool_result` -- Foundry's own `.call_id` is
    reliable for this (re-verified live 2026-08-27; see
    `transport_foundry.py`'s module docstring on why the original spike's
    "no id links a result to its call" note was imprecise).
    """

    kind: Literal["text", "tool_call", "tool_result"]
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    text: str | None = None


@dataclass
class StageResult:
    text: str
    response_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[str] = field(default_factory=list)
    # Tools that ran but answered with an {"error": ...} payload. The tools
    # signal refusal in-band, so a call that happened is not a call that worked,
    # and counting one as the other is how a run reports success having done
    # nothing.
    refused_tools: list[str] = field(default_factory=list)
    # Names of calls that actually succeeded, one entry per successful call.
    # Kept separate from tool_calls/refused_tools rather than derived by
    # set-subtracting them: a model that calls the same tool twice in one
    # turn (a bad argument refused, then a self-corrected retry that
    # succeeds) must have the later success count, and set(tool_calls) -
    # set(refused_tools) collapses both calls to one name and erases it.
    ok_tools: list[str] = field(default_factory=list)
    # Tool name -> its raw JSON result text, whatever it last said (success or
    # refusal). See ds_crew.maf.transport.TurnResult's matching field -- this
    # dataclass is shared/reused by ds_crew.maf, which needs the actual
    # EvaluationBundle/ExplanationBundle payload for its post-hoc guardrails,
    # not just which tools ran.
    tool_results: dict[str, str] = field(default_factory=dict)
    # Ordered, richer version of the above -- see ToolEvent's docstring.
    events: list[ToolEvent] = field(default_factory=list)
    approvals: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)
    transport_retries: int = 0

    def succeeded_tools(self) -> set[str]:
        return set(self.ok_tools)
