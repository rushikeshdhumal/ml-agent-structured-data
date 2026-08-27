"""Behavioral contract for the MAF port, driving the real 9-stage workflow
(`ds_crew.foundry.stages.STAGES`) with a scripted `FakeTransport` instead of a
live Foundry agent. These assertions are the ones that pinned
`ds_crew.foundry.orchestrator`'s retry/nudge/revise/forbidden-tool logic
before this branch replaced it with `ds_crew.maf`; if any of them needed
editing to pass here, the port changed behavior, not just the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ds_crew import settings
from ds_crew.foundry.stages import GATED_TOOLS, STAGES, STAGES_BY_KEY
from ds_crew.maf.executors import (
    MAX_STAGE_ATTEMPTS,
    ForbiddenToolCalled,
    StageDidNotAct,
    build_prompt,
)
from ds_crew.maf.host import describe_checkpoints, drive
from ds_crew.maf.state import PipelineState
from ds_crew.maf.transport import GateAnswer, PendingApproval, TurnResult
from ds_crew.maf.workflow import WORKFLOW_NAME, build_workflow

# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


@dataclass
class FakeTransport:
    """Scripted `StageTransport`: one list of `TurnResult`s per agent name,
    consumed FIFO across that agent's `start`/`answer` calls -- exactly
    mirroring how the same agent name is reused across the `evaluation` and
    `finalize` stages.
    """

    scripts: dict[str, list[TurnResult]]
    calls: list[dict] = field(default_factory=list)

    async def start(self, *, stage, prompt, conversation):
        self.calls.append(
            {"kind": "start", "stage": stage.key, "agent": stage.agent, "prompt": prompt, "conversation": conversation}
        )
        return self.scripts[stage.agent].pop(0)

    async def answer(self, *, stage, conversation, answers: list[GateAnswer]):
        self.calls.append(
            {"kind": "answer", "stage": stage.key, "agent": stage.agent, "conversation": conversation, "answers": answers}
        )
        return self.scripts[stage.agent].pop(0)


def _turn(*, conversation="resp", text="", tool_calls=(), ok=(), refused=(), pending=(), retries=0):
    return TurnResult(
        conversation=conversation,
        text=text,
        input_tokens=10,
        output_tokens=5,
        tool_calls=list(tool_calls),
        ok_tools=list(ok),
        refused_tools=list(refused),
        pending=list(pending),
        transport_retries=retries,
    )


def _pending(tool, *, agent="agent", req_id="req1", arguments="{}"):
    return PendingApproval(id=req_id, tool=tool, arguments=arguments, agent=agent, raw=None)


def _happy_scripts():
    """One successful turn per stage, calling exactly the tool(s) it expects."""
    scripts: dict[str, list[TurnResult]] = {}
    for stage in STAGES:
        turn = _turn(
            conversation=f"{stage.key}-resp",
            text=f"{stage.key} ok",
            tool_calls=list(stage.expects_tools),
            ok=list(stage.expects_tools),
        )
        scripts.setdefault(stage.agent, []).append(turn)
    return scripts


async def _always_approve(req):
    return True, ""


async def _default_verdict(explanation_text):
    return "I approve the recommended model. auto-verdict for tests."


def _run(transport, *, decide=_always_approve, collect_verdict=_default_verdict, run_id="r1"):
    workflow = build_workflow(transport=transport, decide=decide, collect_verdict=collect_verdict, log=lambda _: None)
    return drive(workflow, PipelineState(run_id=run_id))


# ----------------------------------------------------------------------
# Happy path and ordering
# ----------------------------------------------------------------------


async def test_a_full_run_visits_every_stage_in_order():
    transport = FakeTransport(_happy_scripts())
    final = await _run(transport)
    assert list(final.results) == [s.key for s in STAGES]


async def test_finalize_resumes_the_evaluation_stage_conversation():
    """The `resume_from` edge: stage 9 must reattach to stage 7's conversation,
    with stage 8 (a different agent) in between."""
    transport = FakeTransport(_happy_scripts())
    await _run(transport)
    finalize_start = next(c for c in transport.calls if c["kind"] == "start" and c["stage"] == "finalize")
    assert finalize_start["conversation"] == "evaluation-resp"


async def test_backwards_only_dependency_graph():
    """Every stage's `needs` must refer only to earlier stages."""
    seen: list[str] = []
    for stage in STAGES:
        assert set(stage.needs) <= set(seen), f"{stage.key} needs a stage that hasn't run yet"
        seen.append(stage.key)


async def test_gated_tools_matches_the_tools_stages_actually_expect():
    expected_gates = {t for s in STAGES for t in s.expects_tools} & GATED_TOOLS
    assert expected_gates == GATED_TOOLS


# ----------------------------------------------------------------------
# Denial -> revision (the original live bug this session started from)
# ----------------------------------------------------------------------


async def test_a_denied_gate_lets_the_agent_revise_and_try_again():
    scripts = _happy_scripts()
    scripts["model-selector"] = [
        _turn(conversation="m1", pending=[_pending("set_evaluation_metric", agent="model-selector", req_id="mcpr_1")]),
        _turn(conversation="m2", text="Noted, reconsidering."),
        _turn(conversation="m3", pending=[_pending("set_evaluation_metric", agent="model-selector", req_id="mcpr_2")]),
        _turn(conversation="m4", tool_calls=["set_evaluation_metric", "train_candidate_models"],
              ok=["set_evaluation_metric", "train_candidate_models"]),
    ]
    transport = FakeTransport(scripts)

    calls = {"n": 0}

    async def decide_metric_then_approve(req):
        if req.tool != "set_evaluation_metric":
            return True, ""
        calls["n"] += 1
        return (False, "will f1 score be a better metric") if calls["n"] == 1 else (True, "")

    final = await _run(transport, decide=decide_metric_then_approve)

    assert list(final.results) == [s.key for s in STAGES], "the run must continue past the revision"
    assert final.results["model_selection"].denied == ["set_evaluation_metric"]
    assert "set_evaluation_metric" in final.results["model_selection"].succeeded_tools()

    revise_calls = [c for c in transport.calls if c["stage"] == "model_selection" and c["kind"] == "start"
                    and "did not approve" in c["prompt"]]
    assert len(revise_calls) == 1
    assert "set_evaluation_metric" in revise_calls[0]["prompt"]


async def test_persistent_rejection_eventually_stops_the_run():
    """A human declining every revision is a decision, not a stall -- the run
    stops cleanly (GateNotApproved caught inside the executor) rather than
    crashing or looping forever."""
    scripts = _happy_scripts()
    rounds = 1 + MAX_STAGE_ATTEMPTS  # the initial attempt plus every revision
    script = []
    for i in range(rounds):
        script.append(
            _turn(conversation=f"c{2 * i + 1}",
                  pending=[_pending("apply_cleaning_plan", agent="cleaning-strategist", req_id=f"mcpr_{i}")])
        )
        # Settling the denial within the same turn raises no further
        # approval -- only the NEXT stage-level attempt's follow-up prompt
        # (a fresh `start()` continuing the conversation) does that.
        script.append(_turn(conversation=f"c{2 * i + 2}", text="still not applying"))
    scripts["cleaning-strategist"] = script
    transport = FakeTransport(scripts)

    async def refuse(req):
        return (False, "no") if req.tool == "apply_cleaning_plan" else (True, "")

    final = await _run(transport, decide=refuse)
    assert "cleaning" not in final.results
    assert "features" not in final.results, "the run continued past a rejected gate"


# ----------------------------------------------------------------------
# Stall -> nudge
# ----------------------------------------------------------------------


async def test_a_stage_that_only_talks_is_nudged_into_acting():
    scripts = _happy_scripts()
    scripts["eda-analyst"] = [
        _turn(conversation="e1", text="Shall I profile the dataset? Please confirm."),
        _turn(conversation="e2", tool_calls=["eda_summary"], ok=["eda_summary"], text="Profiled."),
    ]
    transport = FakeTransport(scripts)
    final = await _run(transport)

    nudge_calls = [c for c in transport.calls if c["stage"] == "eda" and c["kind"] == "start"
                   and "Do not ask for permission in prose" in c["prompt"]]
    assert len(nudge_calls) == 1
    assert nudge_calls[0]["conversation"] == "e1"
    assert "eda_summary" in final.results["eda"].succeeded_tools()


async def test_a_stage_that_never_acts_fails_the_run_rather_than_reporting_success():
    scripts = _happy_scripts()
    scripts["eda-analyst"] = [_turn(conversation=f"e{i}", text="Awaiting your confirmation.") for i in range(5)]
    transport = FakeTransport(scripts)
    with pytest.raises(StageDidNotAct, match="never called eda_summary"):
        await _run(transport)


# ----------------------------------------------------------------------
# In-band refusal (the tool itself said no)
# ----------------------------------------------------------------------


async def test_an_in_band_refusal_is_not_counted_as_a_successful_call():
    scripts = _happy_scripts()
    scripts["ensembler"] = [
        _turn(conversation="en1", tool_calls=["build_ensemble"], refused=["build_ensemble"],
              text="Could not ensemble.")
    ]
    transport = FakeTransport(scripts)
    with pytest.raises(StageDidNotAct, match="refused the call"):
        await _run(transport)


async def test_a_same_turn_self_correction_is_credited_as_success():
    """The bug fixed in the legacy orchestrator (commit 67728a2): a tool
    refused once, then called again successfully in the same turn, must be
    credited as a success -- not erased because both calls share one name."""
    scripts = _happy_scripts()
    scripts["hpo-tuner"] = [
        _turn(
            conversation="h1",
            tool_calls=["tune_model_hyperparameters", "tune_model_hyperparameters"],
            refused=["tune_model_hyperparameters"],
            ok=["tune_model_hyperparameters"],
            text="Retried with a valid model and it worked.",
        )
    ]
    transport = FakeTransport(scripts)
    final = await _run(transport)
    assert "tune_model_hyperparameters" in final.results["hpo"].succeeded_tools()


# ----------------------------------------------------------------------
# Forbidden tools
# ----------------------------------------------------------------------


def test_evaluation_stage_forbids_finalize_run():
    assert STAGES_BY_KEY["evaluation"].forbidden_tools == ("finalize_run",)


async def test_a_forbidden_tool_stops_the_run_immediately():
    scripts = _happy_scripts()
    scripts["evaluator"] = [
        _turn(conversation="ev1", tool_calls=["evaluate_models", "finalize_run"],
              ok=["evaluate_models", "finalize_run"], text="Evaluated and finalized."),
    ]
    transport = FakeTransport(scripts)
    with pytest.raises(ForbiddenToolCalled, match="belongs to a later turn"):
        await _run(transport)

    # The stage after the violation must never have been reached.
    assert not any(c["stage"] == "explanation" for c in transport.calls)


async def test_a_refused_forbidden_call_does_not_trip_the_guard():
    """If the tool itself refused the premature call, nothing was recorded --
    the guard exists for a call that SUCCEEDED at the wrong time."""
    scripts = _happy_scripts()
    scripts["evaluator"][0] = _turn(
        conversation="ev1", tool_calls=["evaluate_models", "finalize_run"],
        ok=["evaluate_models"], refused=["finalize_run"], text="Evaluated; finalize refused.",
    )
    transport = FakeTransport(scripts)
    final = await _run(transport)
    assert "finalize_run" not in final.results["evaluation"].succeeded_tools()
    assert list(final.results) == [s.key for s in STAGES]


# ----------------------------------------------------------------------
# Critical context precondition
# ----------------------------------------------------------------------


async def test_missing_critical_context_is_checked_before_any_network_call():
    """`features` needs `cleaning`'s output; if cleaning never ran (e.g. this
    stage were driven directly), the precondition must fire before any
    transport call, not after."""
    from ds_crew.maf.executors import StageExecutor

    transport = FakeTransport({"feature-engineer": []})
    executor = StageExecutor(
        STAGES_BY_KEY["features"], transport, _always_approve, index=3, total=9, log=lambda _: None
    )

    class _Ctx:
        async def send_message(self, msg):
            raise AssertionError("should never reach here")

        async def yield_output(self, msg):
            raise AssertionError("should never reach here")

    with pytest.raises(RuntimeError, match="load-bearing"):
        await executor.run_stage(PipelineState(run_id="r1"), _Ctx())
    assert transport.calls == []


# ----------------------------------------------------------------------
# Recovering from a poisoned conversation (live-verified 2026-08-26: a
# transport fault while answering an approval can leave a conversation where
# neither resending the answer nor a plain follow-up prompt is accepted).
# ----------------------------------------------------------------------


async def test_conversation_poisoned_during_approval_restarts_the_stage_fresh():
    """The executor must abandon an unusable conversation and re-run the
    stage from scratch in a new one, rather than crash or get stuck."""
    from ds_crew.maf.executors import StageExecutor
    from ds_crew.maf.transport import ConversationPoisoned

    stage = STAGES_BY_KEY["cleaning"]

    class _PoisonOnceTransport:
        def __init__(self):
            self.start_conversations: list[str | None] = []
            self.answer_calls = 0

        async def start(self, *, stage, prompt, conversation):
            self.start_conversations.append(conversation)
            return _turn(
                conversation="resp",
                pending=(_pending(stage.expects_tools[0], agent=stage.agent),),
            )

        async def answer(self, *, stage, conversation, answers):
            self.answer_calls += 1
            if self.answer_calls == 1:
                raise ConversationPoisoned("stuck both ways")
            return _turn(
                conversation="fresh-resp",
                tool_calls=list(stage.expects_tools),
                ok=list(stage.expects_tools),
            )

    transport = _PoisonOnceTransport()
    executor = StageExecutor(stage, transport, _always_approve, index=2, total=9, log=lambda _: None)

    prompt = build_prompt(stage, PipelineState(run_id="r1"))
    result = await executor._turn(prompt, None, restart_prompt=prompt)

    assert result.succeeded_tools() == set(stage.expects_tools)
    # Both the original attempt and the restart started a brand-new
    # conversation -- the poisoned one was abandoned, not continued.
    assert transport.start_conversations == [None, None]
    assert transport.answer_calls == 2


async def test_a_second_poisoning_in_a_row_is_not_retried_forever():
    """One restart is the budget; a repeat failure is systemic, not a blip,
    and must surface rather than loop."""
    from ds_crew.maf.executors import StageExecutor
    from ds_crew.maf.transport import ConversationPoisoned

    stage = STAGES_BY_KEY["cleaning"]

    class _AlwaysPoisonedTransport:
        async def start(self, *, stage, prompt, conversation):
            return _turn(
                conversation="resp",
                pending=(_pending(stage.expects_tools[0], agent=stage.agent),),
            )

        async def answer(self, *, stage, conversation, answers):
            raise ConversationPoisoned("stuck both ways")

    executor = StageExecutor(
        stage, _AlwaysPoisonedTransport(), _always_approve, index=2, total=9, log=lambda _: None
    )

    with pytest.raises(ConversationPoisoned):
        prompt = build_prompt(stage, PipelineState(run_id="r1"))
        await executor._turn(prompt, None, restart_prompt=prompt)


async def test_restart_after_a_poisoned_follow_up_resends_the_full_stage_prompt():
    """Live-verified 2026-08-27: a `ConversationPoisoned` raised during a
    *follow-up* (revise/nudge) turn must restart with the stage's full
    original prompt, not the short `_REVISE`/`_NUDGE` template text that call
    happened to be sending -- that text references "your last call" and "this
    conversation", neither of which exist in a brand-new one. Getting this
    wrong doesn't crash; it silently confuses the agent into never succeeding,
    which is worse."""
    from ds_crew.maf.executors import StageExecutor
    from ds_crew.maf.transport import ConversationPoisoned

    stage = STAGES_BY_KEY["cleaning"]
    tool = stage.expects_tools[0]

    class _PoisonedDuringRevise:
        def __init__(self):
            self.start_prompts: list[str] = []
            self.answer_calls = 0

        async def start(self, *, stage, prompt, conversation):
            self.start_prompts.append(prompt)
            if len(self.start_prompts) == 3:
                return _turn(conversation="restart-resp", tool_calls=[tool], ok=[tool])
            return _turn(conversation="resp", pending=(_pending(tool, agent=stage.agent),))

        async def answer(self, *, stage, conversation, answers):
            self.answer_calls += 1
            if self.answer_calls == 1:
                return _turn(conversation="resp")  # denial acknowledged, nothing re-proposed yet
            raise ConversationPoisoned("stuck both ways")  # the revised proposal's submission

    decisions = {"count": 0}

    async def deny_once_then_approve(req):
        decisions["count"] += 1
        return (False, "please revise") if decisions["count"] == 1 else (True, "")

    transport = _PoisonedDuringRevise()
    executor = StageExecutor(stage, transport, deny_once_then_approve, index=2, total=9, log=lambda _: None)
    prompt = build_prompt(stage, PipelineState(run_id="r1"))

    result = await executor._until_it_acts(prompt, None)

    assert result.succeeded_tools() == {tool}
    assert len(transport.start_prompts) == 3
    assert transport.start_prompts[1] != prompt  # the follow-up really was the short template
    assert transport.start_prompts[2] == prompt  # but the restart must be the full prompt


# ----------------------------------------------------------------------
# The verdict node
# ----------------------------------------------------------------------


async def test_the_verdict_is_collected_after_the_explanation_and_used_by_finalize():
    scripts = _happy_scripts()
    seen: dict = {}

    async def collect_verdict(explanation_text):
        seen["explanation"] = explanation_text
        return "I approve. Live verdict."

    transport = FakeTransport(scripts)
    await _run(transport, collect_verdict=collect_verdict)

    assert seen["explanation"] == "explanation ok"
    finalize_prompt = next(
        c for c in transport.calls if c["stage"] == "finalize" and c["kind"] == "start"
    )["prompt"]
    assert "I approve. Live verdict." in finalize_prompt


async def test_a_supplied_verdict_skips_the_live_prompt():
    """A caller who already has a decision (--verdict) must never be
    interrupted for one."""
    transport = FakeTransport(_happy_scripts())

    async def must_not_be_called(explanation_text):
        raise AssertionError("collect_verdict must not be called when a verdict is supplied")

    workflow = build_workflow(
        transport=transport, decide=_always_approve, collect_verdict=must_not_be_called, log=lambda _: None
    )
    state = PipelineState(run_id="r1", verdict="Pre-written approval.")
    final = await drive(workflow, state)
    assert list(final.results) == [s.key for s in STAGES]


# ----------------------------------------------------------------------
# Prompt assembly
# ----------------------------------------------------------------------


def test_finalize_embeds_the_explanation_exactly_once():
    from ds_crew.foundry.runner import StageResult

    state = PipelineState(run_id="r1")
    state.results["explanation"] = StageResult(text="the explanation body", response_id="x")
    prompt = build_prompt(STAGES_BY_KEY["finalize"], state)
    assert prompt.count("the explanation body") == 1


def test_context_is_pasted_for_stages_that_need_it():
    from ds_crew.foundry.runner import StageResult

    state = PipelineState(run_id="r1")
    state.results["eda"] = StageResult(text="eda findings", response_id="x")
    prompt = build_prompt(STAGES_BY_KEY["cleaning"], state)
    assert "eda findings" in prompt
    assert "### Output of the eda-analyst stage" in prompt


# ----------------------------------------------------------------------
# Cost reporting
# ----------------------------------------------------------------------


async def test_cost_is_none_unless_both_price_settings_are_configured(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    transport = FakeTransport(_happy_scripts())
    final = await _run(transport)
    assert final.cost_usd() is None


async def test_cost_is_computed_once_both_price_settings_are_set(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", 1.0)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", 2.0)
    transport = FakeTransport(_happy_scripts())
    final = await _run(transport)
    assert final.cost_usd() is not None
    assert final.cost_usd() > 0


# ----------------------------------------------------------------------
# Checkpointing and --resume
# ----------------------------------------------------------------------


class _CrashesAfterNStarts:
    """Like FakeTransport, but raises on the (n+1)th `start()` call -- stands
    in for a process crash so a checkpoint mid-run actually gets exercised."""

    def __init__(self, scripts, *, crash_after: int):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.crash_after = crash_after
        self.starts = 0

    async def start(self, *, stage, prompt, conversation):
        self.starts += 1
        if self.starts > self.crash_after:
            raise RuntimeError("simulated crash")
        return self.scripts[stage.agent].pop(0)

    async def answer(self, *, stage, conversation, answers):
        return self.scripts[stage.agent].pop(0)


async def test_resuming_from_a_checkpoint_continues_rather_than_restarts(tmp_path):
    """The core promise of --resume: a crash partway through leaves a
    checkpoint that a *brand-new* Workflow object (a fresh process, in
    practice) can pick up from -- completed stages are not re-run, and the
    original run_id survives the round trip."""
    from agent_framework import FileCheckpointStorage

    checkpoint_types = ["ds_crew.maf.state:PipelineState", "ds_crew.foundry.runner:StageResult"]
    storage = FileCheckpointStorage(str(tmp_path), allowed_checkpoint_types=checkpoint_types)

    crashing_transport = _CrashesAfterNStarts(_happy_scripts(), crash_after=2)
    workflow1 = build_workflow(
        transport=crashing_transport,
        decide=_always_approve,
        collect_verdict=_default_verdict,
        checkpoint_storage=storage,
        log=lambda _: None,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        await drive(workflow1, PipelineState(run_id="resume-test-run"))

    rows = await describe_checkpoints(storage, workflow_name=WORKFLOW_NAME)
    assert rows, "expected at least one checkpoint to have been saved before the crash"
    furthest = max((r for r in rows if r["stages_done"] is not None), key=lambda r: r["stages_done"])
    assert furthest["run_id"] == "resume-test-run"
    assert furthest["stages_done"] == 2

    # A brand-new storage handle + transport + Workflow object, exactly what a
    # fresh `ds-crew-maf --resume ...` process invocation would build -- proves
    # resume doesn't secretly depend on in-memory state surviving the "crash".
    fresh_scripts = _happy_scripts()
    for stage in STAGES[:2]:
        fresh_scripts[stage.agent].pop(0)
    fresh_transport = FakeTransport(fresh_scripts)
    workflow2 = build_workflow(
        transport=fresh_transport,
        decide=_always_approve,
        collect_verdict=_default_verdict,
        checkpoint_storage=FileCheckpointStorage(str(tmp_path), allowed_checkpoint_types=checkpoint_types),
        log=lambda _: None,
    )
    final = await drive(workflow2, checkpoint_id=furthest["checkpoint_id"])

    assert list(final.results) == [s.key for s in STAGES]
    assert final.run_id == "resume-test-run"
    # The two pre-crash stages must not have been asked to start over.
    assert fresh_transport.calls[0]["stage"] == STAGES[2].key
