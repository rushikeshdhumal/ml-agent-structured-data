"""The seam between a stage executor and however it actually talks to an agent.

`StageTransport` is the one interface `executors.py` depends on. `conversation`
being `None` means "start a new conversation"; a non-`None` handle means
"continue this one" -- the same parameter covers both the nudge/revise
follow-up chain within one stage and the `finalize` stage's `resume_from` reach
back into `evaluation`'s conversation.

`is_refusal` and the transport-retry wrapper are adapted from (not a copy of)
`ds_crew.foundry.runner._is_refusal`/`is_transport_error`: the underlying
content shape differs (see `transport_foundry.py`'s module docstring), but the
detection rule -- an in-band `{"error": ...}` JSON payload means the tool said
no, not that the transport failed -- is identical, because it's a property of
DS-Crew's own tools, not of which API reaches them.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ds_crew.foundry.runner import is_transport_error
from ds_crew.foundry.stages import Stage


@dataclass(frozen=True)
class PendingApproval:
    """A gated tool call waiting on a human. Mirrors `runner.ApprovalRequest`."""

    id: str
    tool: str
    arguments: str
    agent: str
    # The transport's own approval-request object, opaque here. Handed back to
    # the same transport in `GateAnswer.request` so it can build a response
    # without this module needing to know the transport's content types.
    raw: Any = None

    def pretty_arguments(self) -> str:
        try:
            return json.dumps(json.loads(self.arguments), indent=2)
        except (json.JSONDecodeError, TypeError):
            return self.arguments


@dataclass(frozen=True)
class GateAnswer:
    request: PendingApproval
    approved: bool
    reason: str = ""


@dataclass
class TurnResult:
    """One request/response round-trip. Mirrors `runner.StageResult`'s shape."""

    conversation: str
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[str] = field(default_factory=list)
    refused_tools: list[str] = field(default_factory=list)
    ok_tools: list[str] = field(default_factory=list)
    pending: list[PendingApproval] = field(default_factory=list)
    transport_retries: int = 0

    def succeeded_tools(self) -> set[str]:
        return set(self.ok_tools)


class StageTransport(Protocol):
    async def start(
        self, *, stage: Stage, prompt: str, conversation: str | None
    ) -> TurnResult: ...

    async def answer(
        self, *, stage: Stage, conversation: str, answers: list[GateAnswer]
    ) -> TurnResult: ...


class ConversationPoisoned(RuntimeError):
    """`answer()` cannot make progress in this conversation any more, in
    either direction.

    Live-verified 2026-08-26 (feat/maf-workflow): a transport fault while
    submitting an approval answer can leave a conversation where resending
    that same answer is rejected client-side (nothing left to send), *and* a
    plain follow-up prompt in the same conversation is hard-rejected
    server-side with "The following MCP approval requests do not have an
    approval: ...". Neither direction recovers -- only a brand-new
    conversation for the stage can. See `transport_foundry.answer()`'s except
    clause for where this is raised, and `executors.StageExecutor._turn` for
    the restart.
    """


def is_refusal(text: str) -> bool:
    """Did a tool's raw JSON result carry an in-band `{"error": ...}` payload?

    DS-Crew's tools refuse out-of-order and repeat invocations by returning an
    error object rather than raising, so a call that happened is not
    automatically a call that worked.
    """
    if not text:
        return False
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return '"error"' in text
    return isinstance(parsed, dict) and "error" in parsed


# Substrings of the one-shot gated tools' own idempotency guards
# (apply_cleaning_plan, apply_feature_plan, finalize_run -- see
# cleaning_tools.py/feature_tools.py/logging_tools.py). set_evaluation_metric
# has no such guard; it is safely re-settable, so it never matches here.
_ALREADY_DONE_MARKERS = (
    "already been applied for this run",
    "already been called for this run",
)


def is_already_done(text: str) -> bool:
    """Does a refusal actually mean "this already succeeded", not "no"?

    Live-verified 2026-08-26 (feat/maf-workflow): a transport fault while
    submitting an approval answer can lose the response even though the tool
    call itself went on to complete. The nudge that follows then asks the
    agent to call the tool again, which hits exactly this guard. Treating
    that as a fresh refusal would raise `StageDidNotAct` for a stage that
    already succeeded -- the same reasoning the port's live spike used to
    *prove* a prior call had executed server-side (see
    `transport_foundry.py`'s module docstring) applies here too.
    """
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _ALREADY_DONE_MARKERS)


async def with_transport_retries(
    call: Callable[[], Awaitable[Any]],
    *,
    max_retries: int = 4,
    backoff_base_s: float = 3.0,
    log: Callable[[str], None] = print,
) -> tuple[Any, int]:
    """Retry `call` on a transport fault, exactly as `AgentRunner._create` does.

    Foundry re-enumerates the MCP tool list on every invocation, so a single
    dropped `initialize` can fail a call before the model does any work --
    that is a transport fault, not an agent fault, and must be retried rather
    than reported as a failed stage.
    """
    retries = 0
    last: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await call(), retries
        except Exception as exc:  # noqa: BLE001 -- classified immediately below
            if not is_transport_error(exc):
                raise
            last = exc
            retries += 1
            if attempt < max_retries:
                delay = backoff_base_s * (attempt + 1)
                log(
                    f"    ~ transport fault reaching the tool service "
                    f"(retry {attempt + 1}/{max_retries} in {delay:.0f}s)"
                )
                await asyncio.sleep(delay)
    assert last is not None
    raise last


# Decides one approval. Returns (approve, reason).
Decider = Callable[[PendingApproval], Awaitable[tuple[bool, str]]]
