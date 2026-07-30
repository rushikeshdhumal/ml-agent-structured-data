"""Agent/Task/Crew wiring.

Tool `run_id` binding, `output_pydantic` schemas, guardrail functions, and
HITL flags are attached here in Python rather than in the YAML config,
since CrewAI only resolves YAML string references for these fields against
methods on the crew class carrying a matching decorator (`@output_pydantic`,
etc.) -- attaching them directly as constructor kwargs alongside `config=`
is simpler and keeps guardrails/schemas as ordinary importable Python
objects. `config=` values only fill fields not already explicitly set (see
`crewai.utilities.config.process_config`), so passing both is safe.
"""

from __future__ import annotations

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from ds_crew import settings
from ds_crew.guardrails import (
    make_explanation_grounded_guardrail,
    make_finalize_called_guardrail,
    prevent_target_leakage_guardrail,
    prevent_target_modification_guardrail,
    validate_metric_choice_guardrail,
)
from ds_crew.schemas import (
    CleaningPlan,
    EdaReport,
    EnsembleReport,
    EvaluationBundle,
    ExplanationBundle,
    FeatureEngineeringPlan,
    HpoResults,
    Leaderboard,
    MetricChoice,
)
from ds_crew.tools.cleaning_tools import ApplyCleaningPlanTool
from ds_crew.tools.eda_tools import EdaSummaryTool
from ds_crew.tools.ensemble_tools import EnsembleModelsTool
from ds_crew.tools.eval_tools import EvaluateModelsTool
from ds_crew.tools.explain_tools import ExplainModelsTool
from ds_crew.tools.feature_tools import ApplyFeaturePlanTool
from ds_crew.tools.hpo_tools import TuneModelsTool
from ds_crew.tools.logging_tools import FinalizeRunTool
from ds_crew.tools.model_tools import SetMetricTool, TrainCandidateModelsTool


def _build_llm() -> str | LLM:
    """MODEL alone (a bare/provider-prefixed string) covers CrewAI's built-in named
    providers (openai/anthropic/gemini/ollama/deepseek/...). LLM_BASE_URL opts into
    CrewAI's native `custom_openai` routing for any other OpenAI-compatible endpoint
    (e.g. NVIDIA NIM's https://integrate.api.nvidia.com/v1), passing the model id
    through unchanged and using LLM_API_KEY (or the endpoint's own key) for auth.
    """
    if not settings.LLM_BASE_URL:
        return settings.MODEL
    return LLM(
        model=settings.MODEL,
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        custom_openai=True,
    )


@CrewBase
class DsCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, run_id: str):
        self.run_id = run_id

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    @agent
    def eda_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["eda_analyst"],
            llm=_build_llm(),
            tools=[EdaSummaryTool(run_id=self.run_id)],
        )

    @agent
    def cleaning_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config["cleaning_strategist"],
            llm=_build_llm(),
            tools=[ApplyCleaningPlanTool(run_id=self.run_id)],
        )

    @agent
    def feature_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["feature_engineer"],
            llm=_build_llm(),
            tools=[ApplyFeaturePlanTool(run_id=self.run_id)],
        )

    @agent
    def model_selector(self) -> Agent:
        return Agent(
            config=self.agents_config["model_selector"],
            llm=_build_llm(),
            tools=[
                SetMetricTool(run_id=self.run_id),
                TrainCandidateModelsTool(run_id=self.run_id),
            ],
        )

    @agent
    def hpo_tuner(self) -> Agent:
        return Agent(
            config=self.agents_config["hpo_tuner"],
            llm=_build_llm(),
            tools=[TuneModelsTool(run_id=self.run_id)],
        )

    @agent
    def ensembler(self) -> Agent:
        return Agent(
            config=self.agents_config["ensembler"],
            llm=_build_llm(),
            tools=[EnsembleModelsTool(run_id=self.run_id)],
        )

    @agent
    def explainer(self) -> Agent:
        return Agent(
            config=self.agents_config["explainer"],
            llm=_build_llm(),
            tools=[ExplainModelsTool(run_id=self.run_id)],
        )

    @agent
    def evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config["evaluator"],
            llm=_build_llm(),
            tools=[
                EvaluateModelsTool(run_id=self.run_id),
                FinalizeRunTool(run_id=self.run_id),
            ],
        )

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    @task
    def eda_task(self) -> Task:
        return Task(config=self.tasks_config["eda_task"], output_pydantic=EdaReport)

    @task
    def propose_cleaning_task(self) -> Task:
        return Task(
            config=self.tasks_config["propose_cleaning_task"],
            output_pydantic=CleaningPlan,
            guardrail=prevent_target_modification_guardrail,
            human_input=not settings.AUTO_APPROVE,
        )

    @task
    def execute_cleaning_task(self) -> Task:
        return Task(config=self.tasks_config["execute_cleaning_task"])

    @task
    def propose_feature_task(self) -> Task:
        return Task(
            config=self.tasks_config["propose_feature_task"],
            output_pydantic=FeatureEngineeringPlan,
            guardrail=prevent_target_leakage_guardrail,
            human_input=not settings.AUTO_APPROVE,
        )

    @task
    def execute_feature_task(self) -> Task:
        return Task(config=self.tasks_config["execute_feature_task"])

    @task
    def propose_metric_task(self) -> Task:
        return Task(
            config=self.tasks_config["propose_metric_task"],
            output_pydantic=MetricChoice,
            guardrail=validate_metric_choice_guardrail,
            human_input=not settings.AUTO_APPROVE,
        )

    @task
    def set_metric_task(self) -> Task:
        return Task(config=self.tasks_config["set_metric_task"])

    @task
    def model_selection_task(self) -> Task:
        return Task(config=self.tasks_config["model_selection_task"], output_pydantic=Leaderboard)

    @task
    def hpo_task(self) -> Task:
        return Task(config=self.tasks_config["hpo_task"], output_pydantic=HpoResults)

    @task
    def ensembling_task(self) -> Task:
        return Task(config=self.tasks_config["ensembling_task"], output_pydantic=EnsembleReport)

    @task
    def evaluation_task(self) -> Task:
        # Deliberately NOT human-gated: the sign-off gate lives on
        # explanation_task, one step later, so the human reviews held-out
        # metrics and evidence of what the model learned in a single decision
        # rather than being asked to approve on metrics alone and then shown
        # the explanation afterwards.
        return Task(
            config=self.tasks_config["evaluation_task"],
            output_pydantic=EvaluationBundle,
        )

    @task
    def explanation_task(self) -> Task:
        return Task(
            config=self.tasks_config["explanation_task"],
            output_pydantic=ExplanationBundle,
            guardrail=make_explanation_grounded_guardrail(self.run_id),
            human_input=not settings.AUTO_APPROVE,
        )

    @task
    def finalize_task(self) -> Task:
        return Task(
            config=self.tasks_config["finalize_task"],
            guardrail=make_finalize_called_guardrail(self.run_id),
        )

    # ------------------------------------------------------------------
    # Crew
    # ------------------------------------------------------------------

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            memory=False,
            verbose=True,
            max_rpm=settings.MAX_RPM,
        )
