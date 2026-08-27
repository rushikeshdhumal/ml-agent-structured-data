"""Invoke a Foundry-hosted agent and drive its approval pauses.

Agents in a Foundry project are invoked through the OpenAI-compatible Responses
API, not through a separate agents SDK: `AIProjectClient.get_openai_client(
agent_name=...)` returns an `openai.OpenAI` pointed at
`{endpoint}/agents/{name}/endpoint/protocols/openai`. Two details cost real time
to discover and are pinned by tests:

* `model=` must be the agent's **deployment** name, not the agent name. Passing
  the agent name returns `Model must match the agent's model '<deployment>'`.
* Foundry enumerates the MCP tool list at the start of **every** invocation, so
  a single dropped `initialize` fails the whole call before the model does any
  work, surfacing as `tool_user_error: Initialization timed out`. That is a
  transport fault, not an agent fault, and has to be retried rather than
  reported as a failed stage.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
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


@dataclass
class ApprovalRequest:
    """A gated tool call waiting on a human."""

    id: str
    tool: str
    arguments: str
    agent: str

    def pretty_arguments(self) -> str:
        try:
            return json.dumps(json.loads(self.arguments), indent=2)
        except (json.JSONDecodeError, TypeError):
            return self.arguments


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


# Decides one approval. Returns (approve, reason).
ApprovalDecider = Callable[[ApprovalRequest], tuple[bool, str]]


class AgentRunner:
    """Runs one agent conversation to completion, pausing at gated tools."""

    def __init__(
        self,
        project_client: Any,
        *,
        max_transport_retries: int = 4,
        backoff_base_s: float = 3.0,
        max_approval_rounds: int = 12,
        log: Callable[[str], None] = print,
    ) -> None:
        self._project = project_client
        self._clients: dict[str, Any] = {}
        self._max_retries = max_transport_retries
        self._backoff = backoff_base_s
        self._max_rounds = max_approval_rounds
        self._log = log

    def _client_for(self, agent: str) -> Any:
        if agent not in self._clients:
            self._clients[agent] = self._project.get_openai_client(agent_name=agent)
        return self._clients[agent]

    def _create(self, agent: str, **kwargs: Any) -> tuple[Any, int]:
        """One Responses call, retrying transport faults with backoff."""
        retries = 0
        last: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._client_for(agent).responses.create(**kwargs), retries
            except Exception as exc:  # noqa: BLE001 -- classified immediately below
                if not is_transport_error(exc):
                    raise
                last = exc
                retries += 1
                if attempt < self._max_retries:
                    delay = self._backoff * (attempt + 1)
                    self._log(
                        f"    ~ transport fault reaching the tool service "
                        f"(retry {attempt + 1}/{self._max_retries} in {delay:.0f}s)"
                    )
                    time.sleep(delay)
        assert last is not None
        raise last

    def run(
        self,
        *,
        agent: str,
        deployment: str,
        prompt: str,
        decide: ApprovalDecider,
        previous_response_id: str | None = None,
    ) -> StageResult:
        """Send `prompt` to `agent` and settle every approval it raises.

        The Responses API surfaces a gated tool as an `mcp_approval_request`
        output item; the answer goes back as an `mcp_approval_response` input
        item on a follow-up call carrying `previous_response_id`. An agent can
        raise several in one conversation (model-selector gates the metric and
        then trains), so this loops until a response comes back with none.
        """
        kwargs: dict[str, Any] = {"model": deployment, "input": prompt}
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        response, retries = self._create(agent, **kwargs)
        result = StageResult(text="", response_id=response.id, transport_retries=retries)

        for round_no in range(self._max_rounds):
            self._absorb(response, result)
            pending = self._approval_requests(response, agent)
            if not pending:
                break

            answers: list[dict[str, Any]] = []
            for req in pending:
                approve, reason = decide(req)
                (result.approvals if approve else result.denied).append(req.tool)
                answer: dict[str, Any] = {
                    "type": "mcp_approval_response",
                    "approval_request_id": req.id,
                    "approve": approve,
                }
                # Foundry rejects `reason` outright when approve=True:
                # "'reason' cannot be provided when 'approve' is true." (verified
                # live). Only a denial may explain itself.
                if reason and not approve:
                    answer["reason"] = reason
                answers.append(answer)

            response, retries = self._create(
                agent,
                model=deployment,
                input=answers,
                previous_response_id=response.id,
            )
            result.transport_retries += retries
            result.response_id = response.id
        else:
            raise RuntimeError(
                f"{agent} still requesting approvals after {self._max_rounds} rounds; "
                "refusing to loop further."
            )

        return result

    @staticmethod
    def _approval_requests(response: Any, agent: str) -> list[ApprovalRequest]:
        out = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "mcp_approval_request":
                out.append(
                    ApprovalRequest(
                        id=item.id,
                        tool=getattr(item, "name", "?"),
                        arguments=getattr(item, "arguments", "") or "",
                        agent=agent,
                    )
                )
        return out

    @staticmethod
    def _absorb(response: Any, result: StageResult) -> None:
        """Accumulate text, tool calls and tokens across every round.

        Text is appended rather than replaced: the agent explains its plan in
        one round and reports the outcome in a later one, and a human reading
        the transcript needs both halves.
        """
        usage = getattr(response, "usage", None)
        if usage:
            result.input_tokens += getattr(usage, "input_tokens", 0) or 0
            result.output_tokens += getattr(usage, "output_tokens", 0) or 0

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            kind = getattr(item, "type", None)
            if kind == "mcp_call":
                name = getattr(item, "name", "?")
                result.tool_calls.append(name)
                if _is_refusal(item):
                    result.refused_tools.append(name)
                else:
                    result.ok_tools.append(name)
            elif kind == "message":
                for content in getattr(item, "content", []) or []:
                    text = getattr(content, "text", None)
                    if text:
                        chunks.append(text)
        if chunks:
            joined = "\n\n".join(chunks)
            result.text = f"{result.text}\n\n{joined}".strip() if result.text else joined
