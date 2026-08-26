"""The pipeline order, as an explicit graph.

This module exists because Foundry has nowhere to put it. The agent schema has
no `workflow`, `sequential` or ordering primitive, and the `a2a` tool that would
let agents hand off to each other is not offered on gpt-5-family deployments.
Foundry's own workflow construct is retiring on 2026-12-01 and Microsoft advises
against building new ones.

That is not a gap this module apologises for. DS-Crew has ordering *invariants*:
the CrewAI implementation uses `Process.sequential` for correctness, not
preference. Sequencing from code makes the order deterministic and reviewable,
where an LLM-routed handoff would make it probabilistic. The failure mode is not
hypothetical -- driven by hand, the evaluator tried to call `finalize_run`
before the explainer had run, and nothing in the tool layer stopped it, because
an explanation is not a hard prerequisite of a sign-off.

One `Stage` is one agent conversation. Agents that own several consecutive
tasks (`cleaning-strategist`, `feature-engineer`, `model-selector`) work through
them inside a single conversation, exactly as their instructions tell them to.
The `evaluator` is the exception and appears twice: it owns `evaluation_task`
and `finalize_task`, but the explainer runs between them, so its second turn
resumes the first conversation via `resume_from`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Stage:
    """One agent conversation in the pipeline."""

    key: str
    agent: str
    deployment: str
    tasks: tuple[str, ...]
    prompt: str
    # Keys of earlier stages whose output is pasted in as context. Mirrors the
    # `context:` lists in tasks.yaml, filtered to what crosses an agent boundary
    # (anything within one agent is already in its conversation).
    needs: tuple[str, ...] = ()
    # Set when this stage continues an earlier stage's conversation instead of
    # starting a new one.
    resume_from: str | None = None
    # Stages whose context is load-bearing rather than advisory: skipping the
    # paste produces a refusal, not merely a worse answer.
    critical_context: tuple[str, ...] = field(default_factory=tuple)
    # Tools this stage exists to run. A stage that ends without them has not
    # done its job, however plausible its prose, and the orchestrator nudges
    # and then fails rather than moving on.
    expects_tools: tuple[str, ...] = ()
    # Gated tools this stage must NOT run, because they belong to a later,
    # separate turn of the same or another agent. Exists because a per-turn
    # prompt ("do not call finalize_run yet") is a weak instruction against an
    # agent's own persistent job description, which can list the forbidden tool
    # as its natural next step. Observed live: the evaluator called finalize_run
    # during `evaluation_task`, before the explainer had run, recording an
    # irreversible rejection three stages early. The orchestrator checks this
    # immediately after every stage rather than trusting the prompt alone.
    forbidden_tools: tuple[str, ...] = ()


STAGES: tuple[Stage, ...] = (
    Stage(
        key="eda",
        agent="eda-analyst",
        deployment="ds-standard",
        tasks=("eda_task",),
        prompt="Profile the dataset for run {run_id}.",
        expects_tools=("eda_summary",),
    ),
    Stage(
        key="cleaning",
        agent="cleaning-strategist",
        deployment="ds-standard",
        tasks=("propose_cleaning_task", "execute_cleaning_task"),
        prompt=(
            "Run id: {run_id}. Propose a minimal cleaning plan for this run and, once I "
            "approve it, apply it."
        ),
        needs=("eda",),
        expects_tools=("apply_cleaning_plan",),
    ),
    Stage(
        key="features",
        agent="feature-engineer",
        deployment="ds-standard",
        tasks=("propose_feature_task", "execute_feature_task"),
        prompt=(
            "Run id: {run_id}. Propose a feature engineering plan covering every "
            "non-target column and, once I approve it, apply it."
        ),
        needs=("eda", "cleaning"),
        # apply_feature_plan validates column_plans against the columns that
        # exist AFTER cleaning. The EDA report lists the columns that existed
        # before it, so a plan built from EDA alone names dropped columns and is
        # refused with "Unknown columns: [...]".
        critical_context=("cleaning",),
        expects_tools=("apply_feature_plan",),
    ),
    Stage(
        key="model_selection",
        agent="model-selector",
        deployment="ds-standard",
        tasks=("propose_metric_task", "set_metric_task", "model_selection_task"),
        prompt=(
            "Run id: {run_id}. Propose the optimization metric for this run. Once I "
            "approve it, set it and then train the candidate models and report the "
            "leaderboard."
        ),
        needs=("eda", "features"),
        expects_tools=("set_evaluation_metric", "train_candidate_models"),
    ),
    Stage(
        key="hpo",
        agent="hpo-tuner",
        deployment="ds-standard",
        tasks=("hpo_task",),
        # timeout_s is pinned well under Foundry's 100s MCP client timeout,
        # which is not configurable from the server side. The tool's own default
        # (300s, capped at MAX_HPO_TIMEOUT_S) cannot fit and fails every time.
        prompt=(
            "Run id: {run_id}. Tune the top leaderboard candidates. You MUST pass "
            "timeout_s=45 and n_trials=10 so the tool call returns well inside 90 seconds."
        ),
        needs=("model_selection",),
        critical_context=("model_selection",),
        expects_tools=("tune_model_hyperparameters",),
    ),
    Stage(
        key="ensemble",
        agent="ensembler",
        deployment="ds-standard",
        tasks=("ensembling_task",),
        prompt="Run id: {run_id}. Build an ensemble from the strongest candidates.",
        needs=("model_selection", "hpo"),
        expects_tools=("build_ensemble",),
    ),
    Stage(
        key="evaluation",
        agent="evaluator",
        deployment="ds-evaluator",
        tasks=("evaluation_task",),
        prompt=(
            "Run id: {run_id}. Score the tuned candidates and the ensemble on the held-out "
            "test set, then tell me which model you recommend and why. This message asks "
            "only for evaluate_models and your recommendation. finalize_run is a separate "
            "task I will ask for explicitly in a later message, after a human has seen the "
            "explanation stage's output -- it does not exist yet. Do not call finalize_run "
            "in this turn for any reason."
        ),
        needs=("hpo", "ensemble"),
        critical_context=("hpo",),
        expects_tools=("evaluate_models",),
        forbidden_tools=("finalize_run",),
    ),
    Stage(
        key="explanation",
        agent="explainer",
        deployment="ds-standard",
        tasks=("explanation_task",),
        prompt=(
            "Run id: {run_id}. Explain the single model the evaluator recommended. Pass "
            "model_names containing only that model."
        ),
        needs=("evaluation",),
        critical_context=("evaluation",),
        expects_tools=("explain_models",),
    ),
    Stage(
        key="finalize",
        agent="evaluator",
        deployment="ds-evaluator",
        tasks=("finalize_task",),
        prompt=(
            "Here is the explanation stage output.\n\n{explanation}\n\n"
            "{verdict}\n\nRecord this decision now by calling finalize_run for run "
            "{run_id}."
        ),
        needs=("explanation",),
        # Resumes the evaluator's own conversation so the EvaluationBundle it
        # produced is still in context. A fresh conversation would lose it, and
        # finalize_run needs selected_model to name a model that was evaluated.
        resume_from="evaluation",
        critical_context=("explanation",),
        expects_tools=("finalize_run",),
    ),
)

STAGES_BY_KEY = {s.key: s for s in STAGES}

# Tools that pause for human approval. Kept here rather than inferred from the
# agent definitions so the orchestrator can state up front how many gates a run
# will hit, and warn when a gate it expected never arrived.
GATED_TOOLS = frozenset(
    {"apply_cleaning_plan", "apply_feature_plan", "set_evaluation_metric", "finalize_run"}
)
