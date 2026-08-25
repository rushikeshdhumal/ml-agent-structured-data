from __future__ import annotations

import pytest

from ds_crew import settings
from ds_crew.crew import DsCrew, active_llm_provider, foundry_base_url
from ds_crew.schemas import (
    CleaningPlan,
    EdaReport,
    EvaluationBundle,
    ExplanationBundle,
    FeatureEngineeringPlan,
)


@pytest.fixture(scope="module", autouse=True)
def _hermetic_llm_routing():
    """Neutralize provider settings so LLM routing tests don't depend on the
    developer's local `.env`.

    settings.py calls load_dotenv() at import, so a real AZURE_FOUNDRY_ENDPOINT
    in a contributor's .env would otherwise silently change which branch
    _build_llm takes and fail the routing assertions below -- or, with an
    endpoint set but no key, raise from _build_llm and break every structural
    test in this module. CI never has a .env, so this protects local runs.

    Module-scoped and autouse deliberately: `built_crew` is module-scoped too,
    and higher-scoped fixtures are set up first, so a function-scoped fixture
    would be created too late to protect it. Function-level monkeypatching in
    individual tests still overrides this and unwinds correctly afterwards.

    MODEL must be neutralized alongside the URLs, not just the URLs: with
    LLM_BASE_URL cleared, _build_llm returns MODEL as a bare string, and CrewAI
    1.15 resolves bare strings against its *native* provider list only (no
    LiteLLM fallback is installed here). A local MODEL like "z-ai/glm-5.2" is
    not on that list, so clearing the URL without the model raises ImportError
    at crew-build time. "gpt-4o" is a native provider and needs no key to
    construct.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(settings, "AZURE_FOUNDRY_ENDPOINT", None)
        mp.setattr(settings, "AZURE_FOUNDRY_API_KEY", None)
        mp.setattr(settings, "LLM_BASE_URL", None)
        mp.setattr(settings, "MODEL", "gpt-4o")
        yield


@pytest.fixture(scope="module")
def built_crew():
    """One crew built per module, shared by every test that only inspects
    structure.

    Building a crew costs ~7s (CrewAI agent construction, LLM object setup and
    YAML config parsing) and no ML runs at all, so the seven structural tests
    here were paying ~50s purely to rebuild an identical object. Everything this
    fixture is shared by -- agent/task counts, task ordering, guardrail presence,
    process type -- is invariant to AUTO_APPROVE, MODEL and MAX_RPM, so a shared
    instance cannot mask a settings-dependent bug.

    Tests that monkeypatch settings, or that assert on a specific run_id, still
    build their own; sharing this one would defeat what they exist to check.
    """
    return DsCrew(run_id="wiring-test").crew()


def test_crew_builds_eight_agents_and_thirteen_tasks(built_crew):
    assert len(built_crew.agents) == 8
    assert len(built_crew.tasks) == 13


def test_explanation_task_runs_between_evaluation_and_finalize(built_crew):
    # Order is the invariant that makes explanation safe: it may only read
    # X_test after evaluate_models has locked scoring in, and its output must
    # reach the human before finalize records their decision.
    names = [t.name for t in built_crew.tasks]
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


def test_propose_tasks_carry_guardrails(built_crew):
    by_output = {t.output_pydantic: t for t in built_crew.tasks if t.output_pydantic}
    assert by_output[CleaningPlan].guardrail is not None
    assert by_output[FeatureEngineeringPlan].guardrail is not None


def test_finalize_task_carries_guardrail(built_crew):
    finalize_task = next(t for t in built_crew.tasks if t.name == "finalize_task")
    assert finalize_task.guardrail is not None


def test_explanation_task_carries_guardrail(built_crew):
    explanation_task = next(t for t in built_crew.tasks if t.name == "explanation_task")
    assert explanation_task.guardrail is not None


def test_process_is_sequential(built_crew):
    from crewai import Process

    assert built_crew.process == Process.sequential


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


# ----------------------------------------------------------------------
# Azure AI Foundry routing
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "pasted",
    [
        # Every form the Foundry portal actually shows for a resource, plus the
        # trailing-slash and whitespace variants that survive a copy/paste.
        "https://my-res.services.ai.azure.com",
        "https://my-res.services.ai.azure.com/",
        "https://my-res.services.ai.azure.com/models",
        "https://my-res.services.ai.azure.com/models/",
        "https://my-res.services.ai.azure.com/openai",
        "https://my-res.services.ai.azure.com/openai/v1",
        "https://my-res.services.ai.azure.com/openai/v1/",
        "  https://my-res.services.ai.azure.com/openai/v1  ",
    ],
)
def test_foundry_base_url_normalizes_every_portal_endpoint_form(pasted):
    assert foundry_base_url(pasted) == "https://my-res.services.ai.azure.com/openai/v1"


def test_foundry_base_url_handles_azure_openai_resource_host():
    # Azure OpenAI resources use a different host than Foundry Models but the
    # same OpenAI-compatible path.
    assert (
        foundry_base_url("https://my-res.openai.azure.com")
        == "https://my-res.openai.azure.com/openai/v1"
    )


def test_foundry_base_url_strips_only_the_longest_matching_suffix():
    # Guards the ordering in _FOUNDRY_ENDPOINT_SUFFIXES: matching "/openai"
    # before "/openai/v1" would leave a dangling "/v1" on the origin.
    assert "/v1/openai/v1" not in foundry_base_url("https://r.services.ai.azure.com/openai/v1")


def test_foundry_endpoint_routes_all_agents_through_openai_compatible_path(monkeypatch):
    monkeypatch.setattr(settings, "MODEL", "gpt-4o-ds-crew")
    monkeypatch.setattr(settings, "AZURE_FOUNDRY_ENDPOINT", "https://my-res.openai.azure.com")
    monkeypatch.setattr(settings, "AZURE_FOUNDRY_API_KEY", "azure-fake-key-for-testing")
    built = DsCrew(run_id="foundry-test").crew()
    for a in built.agents:
        # MODEL passes through unchanged -- for Azure OpenAI this is the
        # deployment name, which is the whole reason it must not be rewritten.
        assert a.llm.model == "gpt-4o-ds-crew"
        assert a.llm.base_url == "https://my-res.openai.azure.com/openai/v1"
        assert a.llm.api_key == "azure-fake-key-for-testing"


def test_foundry_endpoint_takes_precedence_over_llm_base_url(monkeypatch):
    monkeypatch.setattr(settings, "MODEL", "gpt-4o-ds-crew")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(settings, "LLM_API_KEY", "nvapi-fake-key-for-testing")
    monkeypatch.setattr(settings, "AZURE_FOUNDRY_ENDPOINT", "https://my-res.openai.azure.com")
    monkeypatch.setattr(settings, "AZURE_FOUNDRY_API_KEY", "azure-fake-key-for-testing")
    built = DsCrew(run_id="foundry-precedence-test").crew()
    assert built.agents[0].llm.base_url == "https://my-res.openai.azure.com/openai/v1"
    assert built.agents[0].llm.api_key == "azure-fake-key-for-testing"


def test_foundry_endpoint_without_key_fails_at_build_time(monkeypatch):
    # A missing key must not be allowed to surface as an opaque 401 midway
    # through a run that has already spent tokens.
    monkeypatch.setattr(settings, "AZURE_FOUNDRY_ENDPOINT", "https://my-res.openai.azure.com")
    monkeypatch.setattr(settings, "AZURE_FOUNDRY_API_KEY", None)
    with pytest.raises(ValueError, match="AZURE_FOUNDRY_API_KEY"):
        DsCrew(run_id="foundry-no-key-test").crew()


@pytest.mark.parametrize(
    ("base_url", "foundry_endpoint", "expected"),
    [
        (None, None, "native"),
        ("https://integrate.api.nvidia.com/v1", None, "custom_openai"),
        (None, "https://my-res.openai.azure.com", "azure_foundry"),
        ("https://integrate.api.nvidia.com/v1", "https://my-res.openai.azure.com", "azure_foundry"),
    ],
)
def test_active_llm_provider_labels_each_routing_branch(
    monkeypatch, base_url, foundry_endpoint, expected
):
    # main.py tags this onto the MLflow run; if it disagrees with _build_llm's
    # actual branch, cost comparisons between providers are attributed wrongly.
    monkeypatch.setattr(settings, "LLM_BASE_URL", base_url)
    monkeypatch.setattr(settings, "AZURE_FOUNDRY_ENDPOINT", foundry_endpoint)
    assert active_llm_provider() == expected
