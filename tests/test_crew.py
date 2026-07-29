from __future__ import annotations

from ds_crew import settings
from ds_crew.crew import DsCrew
from ds_crew.schemas import (
    CleaningPlan,
    EdaReport,
    EvaluationBundle,
    ExplanationBundle,
    FeatureEngineeringPlan,
)


def test_crew_builds_eight_agents_and_thirteen_tasks():
    built = DsCrew(run_id="wiring-test").crew()
    assert len(built.agents) == 8
    assert len(built.tasks) == 13


def test_explanation_task_runs_between_evaluation_and_finalize():
    # Order is the invariant that makes explanation safe: it may only read
    # X_test after evaluate_models has locked scoring in, and its output must
    # reach the human before finalize records their decision.
    names = [t.name for t in DsCrew(run_id="order-test").crew().tasks]
    assert names.index("evaluation_task") < names.index("explanation_task")
    assert names.index("explanation_task") < names.index("finalize_task")


def test_tools_are_bound_to_the_given_run_id():
    built = DsCrew(run_id="bound-run-id").crew()
    for a in built.agents:
        for t in a.tools:
            assert t.run_id == "bound-run-id"


def test_propose_and_signoff_tasks_have_output_pydantic_and_human_input(monkeypatch):
    monkeypatch.setattr(settings, "AUTO_APPROVE", False)
    built = DsCrew(run_id="hitl-test").crew()
    by_output = {t.output_pydantic: t for t in built.tasks if t.output_pydantic}
    assert by_output[EdaReport].human_input is False
    assert by_output[CleaningPlan].human_input is True
    assert by_output[FeatureEngineeringPlan].human_input is True
    # The sign-off gate sits on explanation_task, not evaluation_task, so the
    # human approves once with both the held-out metrics and the evidence of
    # what the model learned in front of them -- rather than approving on
    # metrics alone and only then being shown the explanation.
    assert by_output[EvaluationBundle].human_input is False
    assert by_output[ExplanationBundle].human_input is True


def test_auto_approve_disables_human_input(monkeypatch):
    monkeypatch.setattr(settings, "AUTO_APPROVE", True)
    built = DsCrew(run_id="auto-approve-test").crew()
    assert all(t.human_input is False for t in built.tasks)


def test_propose_tasks_carry_guardrails():
    built = DsCrew(run_id="guardrail-test").crew()
    by_output = {t.output_pydantic: t for t in built.tasks if t.output_pydantic}
    assert by_output[CleaningPlan].guardrail is not None
    assert by_output[FeatureEngineeringPlan].guardrail is not None


def test_finalize_task_carries_guardrail():
    built = DsCrew(run_id="finalize-guardrail-test").crew()
    finalize_task = next(t for t in built.tasks if t.name == "finalize_task")
    assert finalize_task.guardrail is not None


def test_explanation_task_carries_guardrail():
    built = DsCrew(run_id="explanation-guardrail-test").crew()
    explanation_task = next(t for t in built.tasks if t.name == "explanation_task")
    assert explanation_task.guardrail is not None


def test_process_is_sequential():
    from crewai import Process

    built = DsCrew(run_id="process-test").crew()
    assert built.process == Process.sequential


def test_plain_model_string_used_when_no_custom_base_url(monkeypatch):
    monkeypatch.setattr(settings, "MODEL", "gpt-4o")
    monkeypatch.setattr(settings, "LLM_BASE_URL", None)
    built = DsCrew(run_id="plain-model-test").crew()
    assert built.agents[0].llm.model == "gpt-4o"


def test_custom_openai_compatible_endpoint_routes_through_base_url(monkeypatch):
    # Mirrors NVIDIA's free NIM endpoints: an OpenAI-compatible API with a model id
    # that isn't one of CrewAI's built-in named providers.
    monkeypatch.setattr(settings, "MODEL", "z-ai/glm-5.2")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(settings, "LLM_API_KEY", "nvapi-fake-key-for-testing")
    built = DsCrew(run_id="nvidia-test").crew()
    for a in built.agents:
        assert a.llm.model == "z-ai/glm-5.2"
        assert a.llm.base_url == "https://integrate.api.nvidia.com/v1"
        assert a.llm.api_key == "nvapi-fake-key-for-testing"


def test_max_rpm_is_passed_through_to_crew(monkeypatch):
    monkeypatch.setattr(settings, "MAX_RPM", 35)
    built = DsCrew(run_id="max-rpm-test").crew()
    assert built.max_rpm == 35
