"""Drive a Foundry-hosted agent via Microsoft Agent Framework's `FoundryAgent`.

Verified live against the real `ds-crew` project 2026-08-26 (see
`maf_workflow_port` memory for the full spike record) before this file was
written -- every mechanic below is proven, not assumed:

* Every content item `agent.run()` returns -- reasoning, tool call, tool
  result, plain text, approval request -- is the **same**
  `agent_framework._types.Content` class, a flexible dataclass with a `.type`
  string discriminator and many always-present nullable fields. There are no
  distinct subclasses to `isinstance`-check; dispatch on `.type`.
* A gated tool call surfaces as `type == "function_approval_request"`, whose
  `.id` is the approval id and whose `.function_call` (itself a `Content`,
  `type == "function_call"`) carries the real `.name`/`.arguments`. Answering
  it is `approval_content.to_function_approval_response(approved=bool)` --
  a method on the content object itself -- fed back into `agent.run()` as a
  plain `list[Content]` (confirmed to genuinely re-execute the tool
  server-side: a direct follow-up call to the tool endpoint returned the
  idempotency guard's "already applied" refusal).
* A tool's own outcome surfaces as `type == "mcp_server_tool_call"` (carries
  `tool_name`, `arguments`, and `call_id`) immediately followed by
  `type == "mcp_server_tool_result"`, whose `.output` is a `list[Content]`
  with one `type == "text"` item whose `.text` is the tool's *exact* raw JSON
  string -- the same shape `ds_crew.foundry.runner._is_refusal` already
  parses. The original 2026-08-26 spike recorded "no id links a result back
  to its call" and fell back to positional correlation (`runner._absorb`
  appends to lists in encounter order rather than keying by id, and
  `_to_turn`'s `tool_results` still does) -- re-checked live 2026-08-27 while
  building `azure_evaluation.py`, and that's imprecise: `.call_id` **is**
  present on both sides and matches (`tool_name`/`.id` are what come back
  `None` on the result side, which is what the original check likely
  observed). `_to_turn`'s `events` list uses the real `call_id`.
* `usage_details` is a plain dict (`input_token_count`/`output_token_count`/
  ...) populated on every turn, tool calls or not.
* `AgentSession()` threads per-agent conversation continuity across
  `agent.run(prompt, session=...)` calls; reattaching to an id from a prior
  session works by constructing `AgentSession(session_id=that_id)`.

Found live during verification (not in the original spike, 2026-08-26): a
transport fault on the *first* submission of an approval answer is not safely
retryable the way a `start()` prompt is, and the failure has two layers:

1. Foundry approval ids are single-use. If the request actually lands but the
   HTTP response back to us is lost, `with_transport_retries` resends the
   identical id; something in the client's own approval-tracking (not
   confirmed server-side, see point 2) treats it as already handled and drops
   it, leaving the OpenAI-compatible chat client with an empty content list --
   it raises `ChatClientInvalidRequestException("Messages are required for
   chat completions")` rather than something recognizable as "already
   handled".
2. That conversation is then a dead end in *both* directions, not just
   unable to resend the approval: a plain follow-up prompt in the same
   conversation was hard-rejected server-side with `400 "The following MCP
   approval requests do not have an approval: ..."` -- Foundry's own Responses
   API still considers the approval genuinely unanswered and refuses any
   further turn until it is. So the client believes it's handled while the
   server believes it's still pending, and neither an answer nor a plain
   prompt can break the deadlock.

`answer()`'s except clause therefore raises `transport.ConversationPoisoned`
rather than guessing at the tool's outcome or trying to nurse the same
conversation along; `executors.StageExecutor._turn` catches it and restarts
the stage in a brand-new conversation, which is safe because DS-Crew's gated
tools are one-shot and refuse a genuine repeat cleanly (`transport.
is_already_done` reads that refusal as success, not failure, in case the
original, now-abandoned attempt secretly *had* gone through).
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_framework import AgentSession
from agent_framework.exceptions import ChatClientInvalidRequestException
from agent_framework.foundry import FoundryAgent
from azure.ai.projects import AIProjectClient

from ds_crew.foundry.runner import ToolEvent
from ds_crew.foundry.stages import Stage
from ds_crew.maf.transport import (
    ConversationPoisoned,
    GateAnswer,
    PendingApproval,
    TurnResult,
    is_already_done,
    is_refusal,
    with_transport_retries,
)


def _tool_result_text(content: Any) -> str:
    output = getattr(content, "output", None)
    if not output:
        return ""
    if isinstance(output, str):
        return output
    parts = [getattr(item, "text", None) for item in output]
    return "\n".join(p for p in parts if p)


def _to_turn(resp: Any, conversation_id: str, retries: int, agent_name: str) -> TurnResult:
    result = TurnResult(conversation=conversation_id, transport_retries=retries)
    usage = getattr(resp, "usage_details", None) or {}
    result.input_tokens = usage.get("input_token_count", 0) or 0
    result.output_tokens = usage.get("output_token_count", 0) or 0

    texts: list[str] = []
    last_call_name = "?"
    for message in resp.messages:
        for content in message.contents:
            kind = getattr(content, "type", None)
            if kind == "text":
                if content.text:
                    texts.append(content.text)
                    result.events.append(ToolEvent(kind="text", text=content.text))
            elif kind == "mcp_server_tool_call":
                last_call_name = getattr(content, "tool_name", None) or "?"
                result.tool_calls.append(last_call_name)
                result.events.append(
                    ToolEvent(
                        kind="tool_call",
                        call_id=getattr(content, "call_id", None),
                        name=last_call_name,
                        arguments=getattr(content, "arguments", None),
                    )
                )
            elif kind == "mcp_server_tool_result":
                text = _tool_result_text(content)
                result.tool_results[last_call_name] = text
                result.events.append(
                    ToolEvent(
                        kind="tool_result",
                        call_id=getattr(content, "call_id", None),
                        name=last_call_name,
                        text=text,
                    )
                )
                if is_already_done(text):
                    # Refused, but only because a one-shot tool's own guard
                    # says this exact action already completed -- that is the
                    # desired state having been reached, not a failure.
                    result.ok_tools.append(last_call_name)
                elif is_refusal(text):
                    result.refused_tools.append(last_call_name)
                else:
                    result.ok_tools.append(last_call_name)
            elif kind == "function_approval_request":
                fc = content.function_call
                result.pending.append(
                    PendingApproval(
                        id=content.id,
                        tool=fc.name,
                        arguments=fc.arguments or "",
                        agent=agent_name,
                        raw=content,
                    )
                )
    if texts:
        result.text = "\n\n".join(texts)
    return result


class FoundryTransport:
    """One `FoundryAgent` per stage agent, one `AgentSession` per conversation."""

    def __init__(
        self,
        *,
        project_endpoint: str,
        credential: Any,
        timeout: float = 300.0,
        log: Any = print,
    ) -> None:
        self._project_endpoint = project_endpoint
        self._credential = credential
        self._timeout = timeout
        self._log = log
        self._project_client = AIProjectClient(
            endpoint=project_endpoint, credential=credential, allow_preview=True
        )
        self._agents: dict[str, FoundryAgent] = {}
        self._agent_versions: dict[str, str] = {}
        self._sessions: dict[str, AgentSession] = {}

    async def _agent_version(self, name: str) -> str:
        if name not in self._agent_versions:
            details = await asyncio.to_thread(self._project_client.agents.get, name)
            self._agent_versions[name] = details.versions.latest.version
        return self._agent_versions[name]

    async def _agent_for(self, name: str) -> FoundryAgent:
        if name not in self._agents:
            version = await self._agent_version(name)
            self._agents[name] = FoundryAgent(
                project_endpoint=self._project_endpoint,
                agent_name=name,
                agent_version=version,
                credential=self._credential,
                # Required for approval requests to surface at all (verified
                # live; matches upstream issue #6652).
                allow_preview=True,
                timeout=self._timeout,
            )
        return self._agents[name]

    def _session_for(self, conversation: str | None) -> AgentSession:
        if conversation is None:
            return AgentSession()
        if conversation not in self._sessions:
            self._sessions[conversation] = AgentSession(session_id=conversation)
        return self._sessions[conversation]

    async def start(self, *, stage: Stage, prompt: str, conversation: str | None) -> TurnResult:
        session = self._session_for(conversation)
        agent = await self._agent_for(stage.agent)
        resp, retries = await with_transport_retries(
            lambda: agent.run(prompt, session=session), log=self._log
        )
        self._sessions[session.session_id] = session
        return _to_turn(resp, session.session_id, retries, stage.agent)

    async def answer(
        self, *, stage: Stage, conversation: str, answers: list[GateAnswer]
    ) -> TurnResult:
        session = self._sessions[conversation]
        agent = await self._agent_for(stage.agent)
        contents = [a.request.raw.to_function_approval_response(approved=a.approved) for a in answers]
        try:
            resp, retries = await with_transport_retries(
                lambda: agent.run(contents, session=session), log=self._log
            )
        except ChatClientInvalidRequestException as exc:
            # See the module docstring: this conversation is now a dead end in
            # both directions (confirmed live -- a plain follow-up prompt in
            # the same conversation gets a hard 400 from Foundry itself,
            # "The following MCP approval requests do not have an approval").
            # Only a fresh conversation for this stage can make progress.
            self._log(
                "    ~ this conversation's pending approval can no longer be "
                "answered or bypassed after a transport fault; restarting the "
                "stage in a new conversation"
            )
            raise ConversationPoisoned(str(exc)) from exc
        return _to_turn(resp, session.session_id, retries, stage.agent)
