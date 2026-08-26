from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ds_crew import settings
from ds_crew.foundry.orchestrator import (
    MAX_STAGE_ATTEMPTS,
    ForbiddenToolCalled,
    PreflightError,
    RunReport,
    StageDidNotAct,
    _build_prompt,
    preflight,
    run_pipeline,
)
from ds_crew.foundry.runner import AgentRunner, ApprovalRequest, is_transport_error
from ds_crew.foundry.stages import GATED_TOOLS, STAGES, STAGES_BY_KEY

# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


def _msg(text):
    return SimpleNamespace(
        type="message", content=[SimpleNamespace(text=text)]
    )


def _mcp_call(name):
    return SimpleNamespace(type="mcp_call", name=name)


def _approval(req_id, name, arguments="{}"):
    return SimpleNamespace(
        type="mcp_approval_request", id=req_id, name=name, arguments=arguments
    )


def _response(rid, output, tokens=(10, 5)):
    return SimpleNamespace(
        id=rid,
        status="completed",
        output=output,
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]),
    )


class FakeResponses:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class FakeAgentClient:
    def __init__(self, script):
        self.responses = FakeResponses(script)


class FakeProject:
    """Hands out one scripted client per agent name."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.clients = {}

    def get_openai_client(self, *, agent_name):
        if agent_name not in self.clients:
            self.clients[agent_name] = FakeAgentClient(self.scripts.get(agent_name, []))
        return self.clients[agent_name]


def _always_approve(req):
    return True, ""


# ----------------------------------------------------------------------
# Transport classification
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Error encountered while enumerating tools from remote server: Initialization timed out",
        "tool_user_error: something",
        "HTTP 504 Gateway Timeout",
    ],
)
def test_transport_faults_are_recognised(message):
    assert is_transport_error(RuntimeError(message))


def test_a_real_agent_error_is_not_treated_as_transport():
    """Retrying a genuine failure would hide it behind four silent attempts."""
    assert not is_transport_error(ValueError("Model must match the agent's model 'ds-standard'"))


def test_transport_faults_are_retried_then_succeed():
    script = [
        RuntimeError("tool_user_error: Initialization timed out"),
        RuntimeError("tool_user_error: Initialization timed out"),
        _response("r1", [_mcp_call("eda_summary"), _msg("done")]),
    ]
    runner = AgentRunner(FakeProject({"a": script}), backoff_base_s=0, log=lambda _: None)
    result = runner.run(agent="a", deployment="d", prompt="p", decide=_always_approve)
    assert result.text == "done"
    assert result.transport_retries == 2


def test_non_transport_errors_propagate_immediately():
    boom = ValueError("Model must match the agent's model 'ds-standard'")
    runner = AgentRunner(FakeProject({"a": [boom]}), backoff_base_s=0, log=lambda _: None)
    with pytest.raises(ValueError):
        runner.run(agent="a", deployment="d", prompt="p", decide=_always_approve)


# ----------------------------------------------------------------------
# Approval loop
# ----------------------------------------------------------------------


def test_approval_request_is_answered_and_conversation_continues():
    script = [
        _response("r1", [_msg("Here is my plan"), _approval("mcpr_1", "apply_cleaning_plan")]),
        _response("r2", [_mcp_call("apply_cleaning_plan"), _msg("Applied.")]),
    ]
    project = FakeProject({"cleaner": script})
    runner = AgentRunner(project, backoff_base_s=0, log=lambda _: None)
    result = runner.run(agent="cleaner", deployment="d", prompt="p", decide=_always_approve)

    second = project.clients["cleaner"].responses.calls[1]
    assert second["previous_response_id"] == "r1"
    assert second["input"] == [
        {"type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": True}
    ]
    assert result.approvals == ["apply_cleaning_plan"]
    # Both halves survive: the plan the human approved and the outcome.
    assert "Here is my plan" in result.text and "Applied." in result.text


def test_approval_never_carries_a_reason_even_if_the_decider_supplies_one():
    """Foundry rejects the call outright: "'reason' cannot be provided when
    'approve' is true." Observed live on the second orchestrated run, after the
    first fix (nudging stalled agents) let a run reach its first real gate.
    """
    script = [
        _response("r1", [_approval("mcpr_1", "apply_cleaning_plan")]),
        _response("r2", [_msg("Applied.")]),
    ]
    project = FakeProject({"cleaner": script})
    runner = AgentRunner(project, backoff_base_s=0, log=lambda _: None)
    runner.run(
        agent="cleaner",
        deployment="d",
        prompt="p",
        decide=lambda req: (True, "looks safe"),  # a decider that over-explains
    )
    sent = project.clients["cleaner"].responses.calls[1]["input"][0]
    assert sent["approve"] is True
    assert "reason" not in sent


def test_denial_is_sent_with_its_reason_and_recorded():
    script = [
        _response("r1", [_approval("mcpr_1", "apply_feature_plan")]),
        _response("r2", [_msg("Understood, I will not apply it.")]),
    ]
    project = FakeProject({"fe": script})
    runner = AgentRunner(project, backoff_base_s=0, log=lambda _: None)
    result = runner.run(
        agent="fe",
        deployment="d",
        prompt="p",
        decide=lambda req: (False, "target would be encoded"),
    )
    sent = project.clients["fe"].responses.calls[1]["input"][0]
    assert sent["approve"] is False
    assert sent["reason"] == "target would be encoded"
    assert result.denied == ["apply_feature_plan"]


def test_multiple_gates_in_one_conversation_are_each_answered():
    """model-selector gates the metric and then trains in the same conversation."""
    script = [
        _response("r1", [_approval("mcpr_1", "set_evaluation_metric")]),
        _response("r2", [_mcp_call("set_evaluation_metric"), _approval("mcpr_2", "finalize_run")]),
        _response("r3", [_mcp_call("finalize_run"), _msg("done")]),
    ]
    runner = AgentRunner(FakeProject({"ms": script}), backoff_base_s=0, log=lambda _: None)
    result = runner.run(agent="ms", deployment="d", prompt="p", decide=_always_approve)
    assert result.approvals == ["set_evaluation_metric", "finalize_run"]


def test_an_endless_approval_loop_is_broken_rather_than_spun_forever():
    script = [_response(f"r{i}", [_approval(f"mcpr_{i}", "finalize_run")]) for i in range(40)]
    runner = AgentRunner(FakeProject({"a": script}), backoff_base_s=0, log=lambda _: None)
    with pytest.raises(RuntimeError, match="after 12 rounds"):
        runner.run(agent="a", deployment="d", prompt="p", decide=_always_approve)


def test_tokens_accumulate_across_approval_rounds():
    script = [
        _response("r1", [_approval("mcpr_1", "finalize_run")], tokens=(100, 20)),
        _response("r2", [_msg("ok")], tokens=(50, 10)),
    ]
    runner = AgentRunner(FakeProject({"a": script}), backoff_base_s=0, log=lambda _: None)
    result = runner.run(agent="a", deployment="d", prompt="p", decide=_always_approve)
    assert (result.input_tokens, result.output_tokens) == (150, 30)


# ----------------------------------------------------------------------
# Stage graph
# ----------------------------------------------------------------------


def test_every_stage_names_a_deployment_and_at_least_one_task():
    for stage in STAGES:
        assert stage.deployment
        assert stage.tasks


def test_stage_dependencies_only_point_backwards():
    """A stage that needs a later stage's output could never run."""
    seen: set[str] = set()
    for stage in STAGES:
        for dep in (*stage.needs, *stage.critical_context):
            assert dep in seen, f"{stage.key} needs {dep}, which has not run yet"
        if stage.resume_from:
            assert stage.resume_from in seen
        seen.add(stage.key)


def test_the_evaluator_resumes_its_own_conversation_to_finalize():
    """finalize_run needs selected_model to name an evaluated model, and the
    EvaluationBundle only exists in the evaluator's first conversation.
    """
    finalize = STAGES_BY_KEY["finalize"]
    assert finalize.resume_from == "evaluation"
    assert finalize.agent == STAGES_BY_KEY["evaluation"].agent


def test_explanation_runs_between_evaluation_and_finalize():
    order = [s.key for s in STAGES]
    assert order.index("evaluation") < order.index("explanation") < order.index("finalize")


def test_hpo_prompt_pins_a_timeout_under_the_foundry_ceiling():
    """Foundry's MCP client aborts at 100s and the tool default is 300s."""
    prompt = STAGES_BY_KEY["hpo"].prompt
    assert "timeout_s=45" in prompt


def test_features_declares_cleaning_as_load_bearing_context():
    assert "cleaning" in STAGES_BY_KEY["features"].critical_context


def test_gated_tools_match_the_documented_four():
    assert GATED_TOOLS == {
        "apply_cleaning_plan",
        "apply_feature_plan",
        "set_evaluation_metric",
        "finalize_run",
    }


# ----------------------------------------------------------------------
# Prompt assembly
# ----------------------------------------------------------------------


def test_prompt_carries_forward_the_needed_stage_outputs():
    report = RunReport(run_id="r1")
    report.results["eda"] = SimpleNamespace(text="EDA SAYS", response_id="x")
    report.results["cleaning"] = SimpleNamespace(text="CLEANING SAYS", response_id="y")
    prompt = _build_prompt(STAGES_BY_KEY["features"], report, "")
    assert "EDA SAYS" in prompt and "CLEANING SAYS" in prompt
    assert "r1" in prompt


def test_finalize_prompt_embeds_the_explanation_once():
    report = RunReport(run_id="r1")
    report.results["explanation"] = SimpleNamespace(text="EXPLANATION BODY", response_id="x")
    prompt = _build_prompt(STAGES_BY_KEY["finalize"], report, "I approve.")
    assert prompt.count("EXPLANATION BODY") == 1
    assert "I approve." in prompt


# ----------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------


def test_preflight_requires_the_project_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "AZURE_FOUNDRY_PROJECT_ENDPOINT", None)
    with pytest.raises(PreflightError, match="AZURE_FOUNDRY_PROJECT_ENDPOINT"):
        preflight()


def test_preflight_requires_a_public_service_url(monkeypatch):
    monkeypatch.setattr(settings, "AZURE_FOUNDRY_PROJECT_ENDPOINT", "https://x/api/projects/y")
    monkeypatch.setattr(settings, "SERVICE_PUBLIC_URL", None)
    with pytest.raises(PreflightError, match="SERVICE_PUBLIC_URL"):
        preflight()


def test_preflight_fails_when_the_tool_service_is_unreachable(monkeypatch):
    """A run that dies at stage 6 leaves applied stages that cannot be redone."""
    monkeypatch.setattr(settings, "AZURE_FOUNDRY_PROJECT_ENDPOINT", "https://x/api/projects/y")
    monkeypatch.setattr(settings, "SERVICE_PUBLIC_URL", "https://nope.invalid")

    def boom(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr("ds_crew.foundry.orchestrator.urllib.request.urlopen", boom)
    with pytest.raises(PreflightError, match="not reachable"):
        preflight(attempts=1, timeout_s=0.1)


# ----------------------------------------------------------------------
# Whole-pipeline behaviour
# ----------------------------------------------------------------------


def _happy_scripts():
    """One response per stage, each actually calling the tools it must call."""
    scripts = {}
    for stage in STAGES:
        items = [_mcp_call(tool) for tool in stage.expects_tools]
        items.append(_msg(f"{stage.key} ok"))
        scripts.setdefault(stage.agent, []).append(_response(f"{stage.key}-resp", items))
    return scripts


def test_a_full_run_visits_every_stage_in_order(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    project = FakeProject(_happy_scripts())
    report = run_pipeline("r1", project_client=project, auto_approve=True, log=lambda _: None)
    assert list(report.results) == [s.key for s in STAGES]


def test_a_denied_gate_lets_the_agent_revise_and_try_again(monkeypatch):
    """The bug this guards against: a human denying a gate with a real reason
    ("will f1 score be a better metric") got the entire nine-stage run killed
    instead of a revised proposal to react to. A denial is feedback, not a
    veto on the whole pipeline -- Ctrl+C is the actual abort mechanism.
    """
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    scripts = _happy_scripts()
    scripts["model-selector"] = [
        _response("m1", [_approval("mcpr_1", "set_evaluation_metric", '{"metric": "roc_auc"}')]),
        _response("m2", [_msg("Noted, reconsidering.")]),
        _response("m3", [_approval("mcpr_2", "set_evaluation_metric", '{"metric": "f1_macro"}')]),
        _response(
            "m4",
            [_mcp_call("set_evaluation_metric"), _mcp_call("train_candidate_models"), _msg("done")],
        ),
    ]
    project = FakeProject(scripts)

    calls = {"n": 0}

    def decide_metric_then_approve(req):
        if req.tool != "set_evaluation_metric":
            return True, ""
        calls["n"] += 1
        return (False, "will f1 score be a better metric") if calls["n"] == 1 else (True, "")

    monkeypatch.setattr(
        "ds_crew.foundry.orchestrator._auto_decider", lambda log: decide_metric_then_approve
    )
    report = run_pipeline("r1", project_client=project, auto_approve=True, log=lambda _: None)

    assert list(report.results) == [s.key for s in STAGES], "the run must continue past the revision"
    assert report.results["model_selection"].denied == ["set_evaluation_metric"]
    assert "set_evaluation_metric" in report.results["model_selection"].succeeded_tools()

    revise_prompt = project.clients["model-selector"].responses.calls[2]["input"]
    assert "did not approve" in revise_prompt
    assert "set_evaluation_metric" in revise_prompt


def test_persistent_rejection_eventually_stops_the_run(monkeypatch):
    """A human declining every revision is a decision, not a stall -- the run
    stops cleanly (no crash, no infinite loop) rather than nudging forever.
    """
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    scripts = _happy_scripts()
    rounds = 1 + MAX_STAGE_ATTEMPTS  # the initial attempt plus every revision
    script = []
    for i in range(rounds):
        script.append(_response(f"c{2 * i + 1}", [_approval(f"mcpr_{i}", "apply_cleaning_plan")]))
        script.append(_response(f"c{2 * i + 2}", [_msg("still not applying")]))
    scripts["cleaning-strategist"] = script
    project = FakeProject(scripts)

    def refuse(req):
        return (False, "no") if req.tool == "apply_cleaning_plan" else (True, "")

    monkeypatch.setattr(
        "ds_crew.foundry.orchestrator._auto_decider", lambda log: refuse
    )
    report = run_pipeline("r1", project_client=project, auto_approve=True, log=lambda _: None)
    assert "cleaning" not in report.results
    assert "features" not in report.results, "the run continued past a rejected gate"


def test_a_stage_that_only_talks_is_nudged_into_acting(monkeypatch):
    """The failure this guard exists for: agents asked permission in prose.

    Instructed to explain a gated action before taking it, the agents stopped
    and waited for a "yes" that an orchestrator never types. No approval request
    was raised because no call was attempted, so a nine-stage run reported
    success and applied nothing but the read-only EDA.
    """
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    scripts = _happy_scripts()
    scripts["eda-analyst"] = [
        _response("e1", [_msg("Shall I profile the dataset? Please confirm.")]),
        _response("e2", [_mcp_call("eda_summary"), _msg("Profiled.")]),
    ]
    project = FakeProject(scripts)
    report = run_pipeline("r1", project_client=project, auto_approve=True, log=lambda _: None)

    nudge = project.clients["eda-analyst"].responses.calls[1]
    assert "Do not ask for permission in prose" in nudge["input"]
    assert nudge["previous_response_id"] == "e1"
    assert "eda_summary" in report.results["eda"].succeeded_tools()


def test_a_stage_that_never_acts_fails_the_run_rather_than_reporting_success(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    scripts = _happy_scripts()
    scripts["eda-analyst"] = [
        _response(f"e{i}", [_msg("Awaiting your confirmation.")]) for i in range(5)
    ]
    project = FakeProject(scripts)
    with pytest.raises(StageDidNotAct, match="never called eda_summary"):
        run_pipeline("r1", project_client=project, auto_approve=True, log=lambda _: None)


def test_an_in_band_refusal_is_not_counted_as_a_successful_call(monkeypatch):
    """Tools refuse out-of-order calls with {"error": ...} and HTTP 200."""
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    scripts = _happy_scripts()
    scripts["ensembler"] = [
        _response(
            "en1",
            [
                SimpleNamespace(
                    type="mcp_call",
                    name="build_ensemble",
                    output='{"error": "Run train_candidate_models first."}',
                ),
                _msg("Could not ensemble."),
            ],
        )
    ]
    project = FakeProject(scripts)
    with pytest.raises(StageDidNotAct, match="refused the call"):
        run_pipeline("r1", project_client=project, auto_approve=True, log=lambda _: None)


def test_a_successful_tool_call_is_not_mistaken_for_a_refusal():
    script = [
        _response(
            "r1",
            [SimpleNamespace(type="mcp_call", name="eda_summary", output='{"run_id": "x"}')],
        )
    ]
    runner = AgentRunner(FakeProject({"a": script}), backoff_base_s=0, log=lambda _: None)
    result = runner.run(agent="a", deployment="d", prompt="p", decide=_always_approve)
    assert result.refused_tools == []
    assert result.succeeded_tools() == {"eda_summary"}


def test_a_same_turn_self_correction_is_credited_as_success():
    """The bug this guards against: succeeded_tools() used to be
    set(tool_calls) - set(refused_tools), a set difference by NAME. An agent
    that called a tool once with a bad argument (refused in-band), then
    self-corrected and called it again successfully in the same turn, saw
    the later success erased by the earlier refusal -- both calls share one
    name, so the set difference reports the tool as never having succeeded.
    Observed live: tune_model_hyperparameters returned a genuine HpoResults
    payload, yet the stage still raised "refused the call".
    """
    script = [
        _response(
            "r1",
            [
                SimpleNamespace(
                    type="mcp_call",
                    name="tune_model_hyperparameters",
                    output='{"error": "[\'xgboost\'] not in top-3 leaderboard candidates"}',
                ),
                SimpleNamespace(
                    type="mcp_call",
                    name="tune_model_hyperparameters",
                    output='{"run_id": "x", "results": [], "warnings": []}',
                ),
                _msg("Retried with a valid model and it worked."),
            ],
        )
    ]
    runner = AgentRunner(FakeProject({"a": script}), backoff_base_s=0, log=lambda _: None)
    result = runner.run(agent="a", deployment="d", prompt="p", decide=_always_approve)
    assert result.succeeded_tools() == {"tune_model_hyperparameters"}
    # Still recorded for diagnostics -- the refusal happened, it just wasn't
    # the last word on this tool.
    assert result.refused_tools == ["tune_model_hyperparameters"]


def test_a_same_turn_self_correction_lets_the_stage_proceed(monkeypatch):
    """Same bug, exercised through run_pipeline rather than the runner
    directly: the hpo stage must not raise StageDidNotAct when the agent's
    only refusal on tune_model_hyperparameters was followed by a successful
    call to the same tool in that turn.
    """
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    scripts = _happy_scripts()
    scripts["hpo-tuner"] = [
        _response(
            "h1",
            [
                SimpleNamespace(
                    type="mcp_call",
                    name="tune_model_hyperparameters",
                    output='{"error": "bad model name"}',
                ),
                SimpleNamespace(
                    type="mcp_call",
                    name="tune_model_hyperparameters",
                    output='{"run_id": "x", "results": [], "warnings": []}',
                ),
                _msg("done"),
            ],
        )
    ]
    project = FakeProject(scripts)
    report = run_pipeline("r1", project_client=project, auto_approve=True, log=lambda _: None)
    assert list(report.results) == [s.key for s in STAGES]
    assert "tune_model_hyperparameters" in report.results["hpo"].succeeded_tools()


def test_evaluation_stage_forbids_finalize_run():
    assert STAGES_BY_KEY["evaluation"].forbidden_tools == ("finalize_run",)


def test_a_forbidden_tool_stops_the_run_immediately(monkeypatch):
    """The failure this guard exists for: the evaluator called finalize_run
    during evaluation_task, three stages before the explainer ran, recording an
    irreversible rejection nobody had reviewed yet. A per-turn "not yet" lost to
    the agent's own persistent job description; the orchestrator must not trust
    the prompt alone.
    """
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    scripts = _happy_scripts()
    scripts["evaluator"] = [
        _response(
            "ev1",
            [
                _mcp_call("evaluate_models"),
                _mcp_call("finalize_run"),  # premature -- explainer has not run
                _msg("Evaluated and finalized."),
            ],
        )
    ]
    project = FakeProject(scripts)
    with pytest.raises(ForbiddenToolCalled, match="belongs to a later turn"):
        run_pipeline("r1", project_client=project, auto_approve=True, log=lambda _: None)

    # Stages after the violation must never have been reached.
    assert "explainer" not in project.clients
    assert len(project.clients["evaluator"].responses.calls) == 1


def test_a_refused_forbidden_call_does_not_trip_the_guard():
    """If the tool itself refused the premature call, nothing was recorded --
    the guard exists for a call that SUCCEEDED at the wrong time, not one the
    tool layer already caught.
    """
    script = [
        _response(
            "r1",
            [
                SimpleNamespace(
                    type="mcp_call", name="finalize_run",
                    output='{"error": "evaluate_models has not been called yet."}',
                ),
                _msg("Could not finalize."),
            ],
        )
    ]
    runner = AgentRunner(FakeProject({"a": script}), backoff_base_s=0, log=lambda _: None)
    result = runner.run(agent="a", deployment="d", prompt="p", decide=_always_approve)
    assert "finalize_run" not in result.succeeded_tools()


def test_every_stage_declares_the_tools_it_must_run():
    for stage in STAGES:
        assert stage.expects_tools, f"{stage.key} would pass without doing anything"


def test_interactive_verdict_is_collected_live_after_the_explanation(monkeypatch):
    """Without this, a verdict had to be written before the run even started,
    on a model nobody had explained yet -- the opposite of what the human gate
    is for. Collecting it live, right before the finalize stage, is the fix.
    """
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    scripts = _happy_scripts()
    seen = {}

    def fake_read_verdict(explanation_text, log):
        seen["explanation"] = explanation_text
        return "I approve. Live verdict."

    project = FakeProject(scripts)
    run_pipeline(
        "r1",
        project_client=project,
        auto_approve=False,
        log=lambda _: None,
        read_verdict=fake_read_verdict,
    )
    assert seen["explanation"] == "explanation ok"

    finalize_prompt = project.clients["evaluator"].responses.calls[-1]["input"]
    assert "I approve. Live verdict." in finalize_prompt


def test_a_supplied_verdict_skips_the_live_prompt(monkeypatch):
    """--verdict is for unattended use; a caller who already has a decision
    should never be interrupted for one.
    """
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    project = FakeProject(_happy_scripts())

    def must_not_be_called(*a, **k):
        raise AssertionError("read_verdict must not be called when a verdict is supplied")

    run_pipeline(
        "r1",
        project_client=project,
        auto_approve=False,
        verdict="Pre-written approval.",
        log=lambda _: None,
        read_verdict=must_not_be_called,
    )


def test_auto_approve_with_no_verdict_records_an_honest_rejection(monkeypatch):
    """Recording a sign-off nobody gave would be worse than an explicit
    rejection -- this is the library-level default, independent of the CLI's
    own copy of the same fallback.
    """
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)
    project = FakeProject(_happy_scripts())
    run_pipeline("r1", project_client=project, auto_approve=True, log=lambda _: None)
    finalize_prompt = project.clients["evaluator"].responses.calls[-1]["input"]
    assert "NOT approved" in finalize_prompt


def test_cost_is_reported_only_when_both_rates_are_configured(monkeypatch):
    report = RunReport(run_id="r")
    report.results["a"] = SimpleNamespace(
        input_tokens=1_000_000, output_tokens=1_000_000, response_id="x", transport_retries=0
    )
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", 3.6)
    assert report.cost_usd() is None

    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", 0.45)
    assert report.cost_usd() == pytest.approx(4.05)


def test_missing_load_bearing_context_stops_before_calling_the_agent(monkeypatch):
    """Better to fail loudly than let a tool refuse halfway through a run."""
    from ds_crew.foundry import orchestrator as orch
    from ds_crew.foundry.stages import Stage

    orphan = Stage(
        key="orphan",
        agent="feature-engineer",
        deployment="ds-standard",
        tasks=("propose_feature_task",),
        prompt="run {run_id}",
        critical_context=("never_ran",),
    )
    monkeypatch.setattr(orch, "STAGES", (orphan,))
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_INPUT", None)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1M_OUTPUT", None)

    project = FakeProject({"feature-engineer": [_response("x", [_msg("should not run")])]})
    with pytest.raises(RuntimeError, match="load-bearing"):
        run_pipeline("r1", project_client=project, auto_approve=True, log=lambda _: None)

    # The agent must never have been contacted.
    assert project.clients == {}


def test_approval_request_pretty_prints_its_arguments():
    req = ApprovalRequest(
        id="i", tool="apply_cleaning_plan", arguments=json.dumps({"run_id": "r", "a": [1]}),
        agent="x",
    )
    assert '"run_id": "r"' in req.pretty_arguments()


def test_pretty_arguments_survives_malformed_json():
    req = ApprovalRequest(id="i", tool="t", arguments="{not json", agent="x")
    assert req.pretty_arguments() == "{not json"
