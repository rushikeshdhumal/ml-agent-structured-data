"""One MAF `Executor` per pipeline stage, plus the human-verdict node.

This is the direct transcription of the pre-MAF hand-written orchestrator's
`_run_stage_until_it_acts` retry loop and `_build_prompt`/`_context_block` --
same precedence, same follow-up prompts, same exceptions -- driving a
`StageTransport` instead of talking to Foundry's Responses API directly, the
way that orchestrator's own `AgentRunner` (since removed, see
`ds_crew.foundry.runner`'s module docstring) used to. See the module
docstrings on `stages.py` and `transport_foundry.py` for why the pipeline is
sequenced in code at all, and what the transport's content shape actually
looks like.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from agent_framework import Executor, WorkflowContext, handler

from ds_crew.foundry.runner import StageResult
from ds_crew.foundry.stages import STAGES_BY_KEY, Stage
from ds_crew.maf.evaluators import (
    find_leakage_suspicions,
    find_ungrounded_model_mentions,
    format_warning_block,
)
from ds_crew.maf.state import PipelineState
from ds_crew.maf.telemetry import get_meter
from ds_crew.maf.transport import ConversationPoisoned, Decider, GateAnswer, StageTransport, TurnResult

# How many follow-up turns a stage gets, whether it's stalling or being told
# "no" -- either way, this is the budget before the executor gives up.
MAX_STAGE_ATTEMPTS = 3

# How many approval rounds one agent turn may raise before this is treated as
# a runaway conversation rather than a legitimately multi-gate turn.
MAX_APPROVAL_ROUNDS = 12

_NUDGE = (
    "You have not completed the tool call yet. Do not ask for permission in prose: "
    "call {tools} now with the arguments you described. Gated tools are paused for "
    "human approval by the platform, which shows me your arguments before anything "
    "runs, so state your reasoning in the same turn as the call rather than waiting "
    "for a reply first."
)

_REVISE = (
    "The human reviewing this did not approve your last call to {tools}. Their "
    "reasoning is already in this conversation, attached to that decision. Revise "
    "your proposal to address it and call {tools} again in this same turn -- do not "
    "just repeat the same arguments, and do not merely respond in prose and wait."
)


class StageDidNotAct(RuntimeError):
    """A stage produced prose but never ran the tool it exists to run."""


class GateNotApproved(RuntimeError):
    """A human declined a gate on every revision attempt offered.

    Distinct from StageDidNotAct: here the agent acted and a human decided,
    every round. That is the system working as designed, not a stall, so the
    workflow stops cleanly with a report rather than crashing on it.
    """


class ForbiddenToolCalled(RuntimeError):
    """A stage ran a gated tool that belongs to a later, separate turn.

    Raised immediately rather than caught by the next stage's precondition
    check, because the call already happened and is often irreversible
    (finalize_run is one-shot server-side).
    """


# Collects the human's free-text sign-off before `finalize`, given the
# explanation stage's text. Returns the verdict string.
VerdictCollector = Callable[[str], Awaitable[str]]


def _context_block(stage: Stage, state: PipelineState) -> str:
    parts: list[str] = []
    for key in stage.needs:
        result = state.results.get(key)
        if result is None or not result.text:
            continue
        parts.append(f"### Output of the {STAGES_BY_KEY[key].agent} stage\n\n{result.text}")
    return "\n\n".join(parts)


def build_prompt(stage: Stage, state: PipelineState) -> str:
    body = stage.prompt.format(
        run_id=state.run_id,
        explanation=(state.results["explanation"].text if "explanation" in state.results else ""),
        verdict=state.verdict,
    )
    # `finalize` embeds the explanation in its own template, so prefixing the
    # generic context block as well would send it twice.
    if stage.key == "finalize":
        return body
    context = _context_block(stage, state)
    return f"{context}\n\n---\n\n{body}" if context else body


def _absorb(turn: TurnResult, result: StageResult) -> None:
    """Accumulate tokens/tool outcomes/text across every approval round.

    Deliberately does not touch `transport_retries` -- that's tracked by the
    caller per request, the same granularity the pre-MAF orchestrator's
    (since-removed) `AgentRunner.run` used.
    """
    result.input_tokens += turn.input_tokens
    result.output_tokens += turn.output_tokens
    result.tool_calls.extend(turn.tool_calls)
    result.refused_tools.extend(turn.refused_tools)
    result.ok_tools.extend(turn.ok_tools)
    result.tool_results.update(turn.tool_results)
    result.events.extend(turn.events)
    if turn.text:
        result.text = f"{result.text}\n\n{turn.text}".strip() if result.text else turn.text


def _merge(base: StageResult, extra: StageResult) -> None:
    base.response_id = extra.response_id
    base.input_tokens += extra.input_tokens
    base.output_tokens += extra.output_tokens
    base.tool_calls.extend(extra.tool_calls)
    base.refused_tools.extend(extra.refused_tools)
    base.ok_tools.extend(extra.ok_tools)
    base.tool_results.update(extra.tool_results)
    base.events.extend(extra.events)
    base.approvals.extend(extra.approvals)
    base.denied.extend(extra.denied)
    base.transport_retries += extra.transport_retries
    if extra.text:
        base.text = f"{base.text}\n\n{extra.text}".strip() if base.text else extra.text


def _check_forbidden(stage: Stage, result: StageResult) -> None:
    hit = set(stage.forbidden_tools) & result.succeeded_tools()
    if hit:
        raise ForbiddenToolCalled(
            f"Stage '{stage.key}' ({stage.agent}) called {', '.join(sorted(hit))}, which "
            f"belongs to a later turn. This may be a one-shot, irreversible action already "
            f"recorded against the run. The agent said:\n\n{result.text[:800]}"
        )


class StageExecutor(Executor):
    """One agent conversation in the pipeline. One instance per `Stage`."""

    def __init__(
        self,
        stage: Stage,
        transport: StageTransport,
        decide: Decider,
        *,
        index: int,
        total: int,
        is_last: bool = False,
        log: Any = print,
    ) -> None:
        super().__init__(id=stage.key)
        self._stage = stage
        self._transport = transport
        self._decide = decide
        self._index = index
        self._total = total
        self._is_last = is_last
        self._log = log
        # Phase 11: the same refusal/denial/retry counts run_stage already
        # logs to console, also as OTel counters -- the early-warning signal
        # that a model or a tool contract is degrading, per-stage/agent, in
        # Application Insights' Metrics blade. Safe no-ops if
        # setup_observability() was never called (see telemetry.get_meter()).
        meter = get_meter()
        self._refused_counter = meter.create_counter(
            "ds_crew.stage.tool_refused",
            description="Tool calls a stage's tools refused in-band (an {\"error\": ...} payload).",
        )
        self._denied_counter = meter.create_counter(
            "ds_crew.stage.tool_denied",
            description="Gated-tool calls a human declined before the agent revised and retried.",
        )
        self._retry_counter = meter.create_counter(
            "ds_crew.stage.transport_retries",
            description="Transport-fault retries absorbed while driving a stage's agent conversation.",
        )

    @handler
    async def run_stage(
        self, state: PipelineState, ctx: WorkflowContext[PipelineState, PipelineState]
    ) -> None:
        stage = self._stage

        missing = [k for k in stage.critical_context if k not in state.results]
        if missing:
            raise RuntimeError(
                f"Stage '{stage.key}' needs output from {missing}, which did not run. "
                "That context is load-bearing, not advisory: the tool refuses without it."
            )

        prior = state.conversations.get(stage.resume_from) if stage.resume_from else None
        prompt = build_prompt(stage, state)

        started = time.time()
        self._log(f"[{self._index}/{self._total}] {stage.agent:<20} ({', '.join(stage.tasks)})")

        try:
            result = await self._until_it_acts(prompt, prior)
        except GateNotApproved as exc:
            self._log(f"    REJECTED: {exc} -- stopping.")
            await ctx.yield_output(state.finish())
            return

        state = state.with_result(stage.key, result, result.response_id)

        tools = ", ".join(sorted(result.succeeded_tools())) or "none"
        self._log(
            f"    done in {time.time() - started:5.1f}s | tools: {tools} | "
            f"tokens {result.input_tokens}/{result.output_tokens}"
        )
        attrs = {"stage": stage.key, "agent": stage.agent}
        if result.refused_tools:
            self._log(f"    refused: {', '.join(sorted(set(result.refused_tools)))}")
            self._refused_counter.add(len(result.refused_tools), attrs)
        if result.denied:
            revised = sorted(set(result.denied))
            self._log(f"    (revised after the human first declined: {', '.join(revised)})")
            self._denied_counter.add(len(result.denied), attrs)
        if result.transport_retries:
            self._retry_counter.add(result.transport_retries, attrs)

        if self._is_last:
            await ctx.yield_output(state.finish())
        else:
            await ctx.send_message(state)

    async def _turn(
        self,
        prompt: str,
        conversation: str | None,
        *,
        restart_prompt: str,
        _restarted: bool = False,
    ) -> StageResult:
        """One full agent turn, settling every approval it raises.

        Same shape as the pre-MAF orchestrator's (since-removed) `AgentRunner.run`:
        an agent can raise several approvals in one conversation, so this loops
        until a turn comes back with none.

        `restart_prompt` is always the stage's full original prompt, never
        `prompt` itself -- `prompt` is whatever this particular call is
        sending, which on a follow-up round is just the short `_NUDGE`/
        `_REVISE` template text. That text is meaningless as the *opening*
        message of the brand-new conversation a `ConversationPoisoned` restart
        creates (verified live 2026-08-26: it references "your last call" and
        "this conversation", neither of which exist yet), so a restart must
        always resend the full prompt regardless of which call triggered it.
        """
        turn = await self._transport.start(stage=self._stage, prompt=prompt, conversation=conversation)
        result = StageResult(text="", response_id=turn.conversation, transport_retries=turn.transport_retries)

        for _ in range(MAX_APPROVAL_ROUNDS):
            _absorb(turn, result)
            if not turn.pending:
                break

            answers: list[GateAnswer] = []
            for req in turn.pending:
                approve, reason = await self._decide(req)
                (result.approvals if approve else result.denied).append(req.tool)
                answers.append(GateAnswer(request=req, approved=approve, reason=reason))

            try:
                turn = await self._transport.answer(
                    stage=self._stage, conversation=result.response_id, answers=answers
                )
            except ConversationPoisoned:
                if _restarted:
                    raise
                # See `transport.ConversationPoisoned`'s docstring: this
                # conversation is a dead end in both directions after a
                # transport fault on the approval answer. Abandon it and
                # re-run the whole stage prompt fresh, exactly once -- a
                # second poisoning in a row means something systemic (not a
                # one-off fault), which should surface, not loop forever.
                self._log(
                    f"    ~ {self._stage.agent}'s conversation is unusable after a "
                    "transport fault; starting over in a new conversation"
                )
                return await self._turn(
                    restart_prompt, None, restart_prompt=restart_prompt, _restarted=True
                )
            result.transport_retries += turn.transport_retries
            result.response_id = turn.conversation
        else:
            raise RuntimeError(
                f"{self._stage.agent} still requesting approvals after {MAX_APPROVAL_ROUNDS} "
                "rounds; refusing to loop further."
            )

        return result

    async def _until_it_acts(self, prompt: str, prior: str | None) -> StageResult:
        """Run a stage, following up when it stalls or is turned down, until it acts.

        Two different silences look alike (no expected tool call yet) but need
        different follow-ups. An agent that stalls needs telling to act instead
        of asking. An agent whose proposal was denied already has the human's
        reason in its own conversation and needs telling to revise, not to
        repeat the same call or wait for a different answer to the same one.
        """
        stage = self._stage
        result = await self._turn(prompt, prior, restart_prompt=prompt)
        _check_forbidden(stage, result)

        expected = set(stage.expects_tools)
        if not expected:
            return result

        for attempt in range(MAX_STAGE_ATTEMPTS):
            missing = expected - result.succeeded_tools()
            if not missing:
                return result

            # Scoped to `missing`, not `expected`: a tool refused once but
            # called again successfully in the same turn (self-correction) is
            # already excluded from `missing` above and must not trip this
            # hard stop.
            refused = sorted(set(result.refused_tools) & missing)
            if refused:
                raise StageDidNotAct(
                    f"Stage '{stage.key}': {', '.join(refused)} refused the call. "
                    f"The agent's account was:\n\n{result.text[:800]}"
                )

            denied = sorted(set(result.denied) & missing)
            if denied:
                self._log(
                    f"    ~ human denied {', '.join(denied)}; asking the agent to revise "
                    f"({attempt + 1}/{MAX_STAGE_ATTEMPTS})"
                )
                follow_up_prompt = _REVISE.format(tools=", ".join(sorted(missing)))
            else:
                self._log(
                    f"    ~ agent stalled without calling {', '.join(sorted(missing))}; nudging "
                    f"({attempt + 1}/{MAX_STAGE_ATTEMPTS})"
                )
                follow_up_prompt = _NUDGE.format(tools=", ".join(sorted(missing)))

            follow_up = await self._turn(follow_up_prompt, result.response_id, restart_prompt=prompt)
            _check_forbidden(stage, follow_up)
            _merge(result, follow_up)

        missing = expected - result.succeeded_tools()
        if set(result.denied) & missing:
            raise GateNotApproved(
                f"Stage '{stage.key}': the human did not approve {', '.join(sorted(missing))} "
                f"after {MAX_STAGE_ATTEMPTS} revisions. Nothing was applied."
            )
        raise StageDidNotAct(
            f"Stage '{stage.key}' never called {', '.join(sorted(missing))} after "
            f"{MAX_STAGE_ATTEMPTS} attempts. Nothing was applied, so the run cannot continue. "
            f"The agent's last message was:\n\n{result.text[:800]}"
        )


class GroundingCheckExecutor(Executor):
    """Deterministic safety-net checks between `explanation` and the human
    verdict -- see `ds_crew.maf.evaluators` for what's actually checked and
    why it's deterministic rather than an LLM judge.

    A separate workflow node, not folded into `StageExecutor` or
    `HumanVerdictExecutor`, for the same reason `HumanVerdictExecutor`
    already is one: this is a first-class part of the pipeline's shape,
    visible in `WorkflowViz`, not a hidden side effect of a stage class doing
    something else.

    Findings are appended to `explanation`'s own `StageResult.text` rather
    than the agent's narration being policed for whether it already mentions
    them -- that text is the one place both `interactive_verdict_collector`
    (what a human reads before approving) and the `finalize` stage's prompt
    (`{explanation}` in `build_prompt`) already consume, so one injection
    point reaches both, regardless of what the agent's own prose said.
    """

    def __init__(self, log: Any = print) -> None:
        super().__init__(id="grounding_check")
        self._log = log

    @handler
    async def check(self, state: PipelineState, ctx: WorkflowContext[PipelineState]) -> None:
        evaluation = state.results.get("evaluation")
        explanation = state.results.get("explanation")

        findings: list[str] = []
        if evaluation is not None:
            findings += find_leakage_suspicions(evaluation.tool_results.get("evaluate_models"))
        if explanation is not None:
            findings += find_ungrounded_model_mentions(
                explanation.text,
                evaluation.tool_results.get("evaluate_models") if evaluation else None,
                explanation.tool_results.get("explain_models"),
            )

        if findings and explanation is not None:
            self._log("    ~ automated safety check flagged:")
            for finding in findings:
                self._log(f"      - {finding}")
            updated = replace(
                explanation, text=f"{explanation.text}\n\n{format_warning_block(findings)}"
            )
            state = state.with_result("explanation", updated, updated.response_id)

        await ctx.send_message(state)


class HumanVerdictExecutor(Executor):
    """Collects the human's free-text sign-off before `finalize`.

    A separate workflow node rather than folded into the `finalize` stage, so
    it renders as its own box in a `WorkflowViz` diagram: the human gate is a
    first-class part of the pipeline's shape, not a hidden branch inside an
    agent stage.
    """

    def __init__(self, collect_verdict: VerdictCollector) -> None:
        super().__init__(id="human_verdict")
        self._collect_verdict = collect_verdict

    @handler
    async def collect(self, state: PipelineState, ctx: WorkflowContext[PipelineState]) -> None:
        if state.verdict:
            # Already supplied (--verdict, or the unattended default) -- never
            # interrupt a caller who already has a decision.
            await ctx.send_message(state)
            return
        explanation_text = state.results["explanation"].text
        verdict = await self._collect_verdict(explanation_text)
        await ctx.send_message(state.with_verdict(verdict))
