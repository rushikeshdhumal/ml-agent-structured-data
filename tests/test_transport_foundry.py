"""Regression tests for FoundryTransport's handling of a lost approval-answer
response, and for reading a one-shot tool's idempotency guard as success.

Live-verified 2026-08-26 against the real `ds-crew` project (see the module
docstring on `transport_foundry.py`): a transport fault on the *first*
submission of an approval answer can mean the request landed at Foundry and
was applied, but the HTTP response back to us was lost. `with_transport_retries`
then resends the identical (single-use) approval id; the client's own
approval-tracking treats it as already handled and drops it, leaving the SDK
with nothing to build a request from -- it raises
`ChatClientInvalidRequestException("Messages are required for chat
completions")`. That conversation then turns out to be a dead end in *both*
directions (also live-verified): a plain follow-up prompt in the same
conversation gets a hard 400 from Foundry itself ("The following MCP approval
requests do not have an approval"), so `answer()` must not try to nurse the
same conversation along -- it raises `ConversationPoisoned` and leaves the
restart to `StageExecutor._turn` (see `test_maf_workflow.py`).
"""

from __future__ import annotations

from typing import Any

import pytest
from agent_framework.exceptions import ChatClientInvalidRequestException

from ds_crew.foundry.stages import STAGES_BY_KEY
from ds_crew.maf.transport import ConversationPoisoned, GateAnswer, PendingApproval, is_already_done
from ds_crew.maf.transport_foundry import FoundryTransport, _to_turn


class _FakeSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


class _FakeApprovalContent:
    def to_function_approval_response(self, *, approved: bool) -> Any:
        return {"approved": approved}


class _RaisingAgent:
    """Mimics the SDK raising because a resent approval id was already resolved."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise self._exc


def _make_transport() -> FoundryTransport:
    transport = FoundryTransport.__new__(FoundryTransport)
    transport._log = lambda *a, **k: None
    transport._sessions = {}
    transport._agents = {}
    return transport


def _pending(tool: str = "apply_cleaning_plan") -> PendingApproval:
    return PendingApproval(
        id="mcpr_1", tool=tool, arguments="{}", agent="cleaning-strategist", raw=_FakeApprovalContent()
    )


@pytest.mark.parametrize("approved", [True, False])
async def test_answer_raises_conversation_poisoned_instead_of_crashing_or_guessing(approved):
    stage = STAGES_BY_KEY["cleaning"]
    transport = _make_transport()
    transport._sessions["conv-1"] = _FakeSession("conv-1")
    fake_agent = _RaisingAgent(
        ChatClientInvalidRequestException("Messages are required for chat completions")
    )
    transport._agents[stage.agent] = fake_agent
    answers = [GateAnswer(request=_pending(), approved=approved, reason="")]

    # No crash with an unrecognizable exception, and no guess at the outcome
    # either (see module docstring: a plain follow-up in this same
    # conversation is independently known to be rejected server-side too) --
    # the caller (StageExecutor._turn) is expected to restart in a fresh
    # conversation, not this transport.
    with pytest.raises(ConversationPoisoned):
        await transport.answer(stage=stage, conversation="conv-1", answers=answers)

    assert fake_agent.calls == 1  # not a transport error, so no blind retry


async def test_answer_still_raises_for_an_unrelated_invalid_request_error():
    """The recovery is scoped to this one exception type -- anything else must
    still surface, not be silently swallowed as "already applied"."""
    stage = STAGES_BY_KEY["cleaning"]
    transport = _make_transport()
    transport._sessions["conv-1"] = _FakeSession("conv-1")
    fake_agent = _RaisingAgent(RuntimeError("some other failure"))
    transport._agents[stage.agent] = fake_agent
    answers = [GateAnswer(request=_pending(), approved=True, reason="")]

    with pytest.raises(RuntimeError, match="some other failure"):
        await transport.answer(stage=stage, conversation="conv-1", answers=answers)


@pytest.mark.parametrize(
    "text",
    [
        '{"error": "Cleaning has already been applied for this run. Do not call it again."}',
        '{"error": "Feature engineering has already been applied for this run. Do not call it again."}',
        '{"error": "finalize_run has already been called for this run. Do not call it again."}',
    ],
)
def test_is_already_done_recognizes_every_one_shot_guard_message(text):
    assert is_already_done(text) is True


def test_is_already_done_does_not_match_a_genuine_refusal():
    assert is_already_done('{"error": "Unknown columns: [\'bogus\']"}') is False
    assert is_already_done("") is False


class _FakeToolResult:
    type = "mcp_server_tool_result"

    def __init__(self, text: str) -> None:
        self.output = [_FakeTextContent(text)]


class _FakeTextContent:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeToolCall:
    type = "mcp_server_tool_call"
    tool_name = "apply_cleaning_plan"


class _FakeMessage:
    def __init__(self, contents: list[Any]) -> None:
        self.contents = contents


class _FakeResponse:
    def __init__(self, messages: list[_FakeMessage]) -> None:
        self.messages = messages
        self.usage_details = {"input_token_count": 1, "output_token_count": 1}


def test_to_turn_reads_a_resent_idempotency_refusal_as_success_not_refusal():
    """A follow-up nudge that re-hits a one-shot tool's own guard (because the
    original call actually succeeded but our earlier response was lost) must
    not be misread as a genuine refusal -- that would raise `StageDidNotAct`
    for a stage that already completed."""
    resp = _FakeResponse(
        [
            _FakeMessage(
                [
                    _FakeToolCall(),
                    _FakeToolResult(
                        '{"error": "Cleaning has already been applied for this run. Do not call it again."}'
                    ),
                ]
            )
        ]
    )

    turn = _to_turn(resp, "conv-1", retries=0, agent_name="cleaning-strategist")

    assert turn.ok_tools == ["apply_cleaning_plan"]
    assert turn.refused_tools == []
