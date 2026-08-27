"""On-demand evaluation of a completed run via the Azure AI Evaluation SDK.

Separate from `ds_crew.maf.evaluators` (Phase 9's deterministic guardrails)
on purpose: that module is dependency-free by design and runs inline, inside
every live pipeline run. This module pulls in `azure-ai-evaluation` and
makes real, billed LLM-judge calls, so it only ever runs when explicitly
asked for (`ds-crew-maf --evaluate <checkpoint-id>`), never automatically.
"Continuous evaluation" (wiring this into every run) is a deliberately
separate, larger decision -- not made here.

Three evaluators, each answering a different question a deterministic check
can't:

* `GroundednessEvaluator` -- is the `explanation` stage's narration actually
  substantiated by `explain_models`' structured report? The LLM-judged
  upgrade path `ds_crew.tools.explain_tools`' docstring already points to.
* `TaskAdherenceEvaluator` -- did a stage's actions actually match its task
  instructions? The `finalize_run` blank-`selected_model` bug (fixed in
  commit `dc0a5e4`) is exactly the failure class this exists to catch --
  automatically, not only when someone happens to watch a live run crash.
* `ToolCallAccuracyEvaluator` -- were a stage's tool calls relevant and
  correctly argued? This pipeline is entirely tool-call-driven, so this
  maps directly onto its core design.

Both TaskAdherence and ToolCallAccuracy need far richer input than
Groundedness: chat-format `query`/`response` message lists, with tool calls
and their results as typed, correlated content blocks -- see `ToolEvent` on
`ds_crew.foundry.runner.StageResult` for where that comes from.
`build_conversation()` is where that reconstruction happens.

Reads `ds_crew.service.registry` for tool schemas (`tool_definitions`) --
a deliberate, narrow exception to the rest of `ds_crew.maf` staying
decoupled from `ds_crew.tools`. That decoupling exists for the live
orchestration hot path (`ds_crew.maf` drives Foundry agents that reach
tools over HTTP/MCP, never in-process); this module isn't part of a live
run at all, so reading the tool registry directly -- rather than hand-
duplicating its schemas here, which would drift -- is the right call.
`registry.py` imports only `pydantic` + `ds_crew.tools.*`, no `service`
extra required just to read schemas off the classes.

Known gap, found live 2026-08-27, not yet closed: `build_conversation()`'s
`response` never represents a gated tool's human-approval step (there is no
`ToolEvent` kind for `function_approval_request`/its answer -- see
`transport_foundry._to_turn()`), so `TaskAdherenceEvaluator` sees a
gated tool called with no visible approval in between and can score a
stage that *was* correctly approved (out of band, via Foundry's structured
approval mechanism) as a procedural failure ("applied without waiting for
approval"). Confirmed live: exactly this pattern on `cleaning` and
`model_selection`, both real approved gates. Treat a low
`task_adherence` score on a gated stage with that in mind -- it is not
necessarily a real defect. (The same run's `features` stage failure was
*not* this artifact: the judge caught the narration claiming drop-first
one-hot encoding while the tool's actual output shows full one-hot, three
indicator columns -- a genuine narration/reality mismatch, exactly the
class of thing this module exists to catch.)
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ds_crew import settings
from ds_crew.foundry.runner import StageResult, ToolEvent
from ds_crew.foundry.stages import STAGES_BY_KEY, Stage
from ds_crew.maf.executors import build_prompt
from ds_crew.maf.state import PipelineState


def judge_model_config() -> dict[str, Any]:
    """`AzureOpenAIModelConfiguration` for the shared judge deployment.

    Points at the same `ds-crew-resource` account the pipeline's own agents
    run on, via its plain Azure-OpenAI-compatible endpoint (not the Foundry
    Agents surface) -- see `settings.AZURE_OPENAI_ENDPOINT`'s comment on why
    those are genuinely different hostnames.

    Deliberately carries no `credential` key, even though
    `AzureOpenAIModelConfiguration`'s type hints allow one -- live-caught
    2026-08-27: `azure-ai-evaluation==1.18.3`'s own config validator does
    `isinstance(v, typing.Any)` on that field, which Python's `isinstance`
    cannot do (`TypeError: typing.Any cannot be used with isinstance()`);
    the validator swallows that and falls through to a second check
    (`OpenAIModelConfiguration`, the non-Azure shape) that then rejects
    every *other* key as unknown, so any config with a `credential` field
    is rejected outright. Entra auth still works -- `credential` just has
    to reach the evaluator through its own separate `credential=` kwarg
    (see `_evaluator_kwargs()`), which this bug's fallback path never touches.
    """
    if not settings.AZURE_OPENAI_ENDPOINT:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is not set -- required for --evaluate's judge model. "
            "See .env.example."
        )
    return {
        "azure_endpoint": settings.AZURE_OPENAI_ENDPOINT,
        "azure_deployment": settings.AZURE_OPENAI_JUDGE_DEPLOYMENT,
        # Pinned rather than left to the SDK's own default -- verified
        # working live 2026-08-27 against ds-standard (Entra auth, plain
        # chat completions) with this exact version.
        "api_version": "2024-10-21",
    }


def _evaluator_kwargs() -> dict[str, Any]:
    """`model_config` + `credential` + `is_reasoning_model`, the arguments
    every evaluator class actually wants -- see `judge_model_config()`'s
    docstring for why `credential` cannot live inside the config dict
    itself. `is_reasoning_model=True` because `ds-standard` (the default
    judge deployment) is a gpt-5-family reasoning model: without this the
    evaluator's own prompty templates request `max_tokens`, which such
    models reject outright (`max_completion_tokens` is required instead) --
    live-caught 2026-08-27.
    """
    from azure.identity import DefaultAzureCredential

    return {
        "model_config": judge_model_config(),
        "credential": DefaultAzureCredential(),
        "is_reasoning_model": True,
    }


def _text_message(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _parsed_arguments(raw: str | None) -> dict[str, Any]:
    """Always a dict -- the SDK's tool_call content validator requires one
    (`_validate_dict_field`), so a parse failure falls back to `{}` rather
    than the raw string, which would fail validation just as hard as no
    arguments at all.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _response_messages(events: Iterable[ToolEvent]) -> list[dict[str, Any]]:
    """Reconstruct a chat-format `response` from a stage's ordered events.

    One assistant message per text/tool_call event (matching how they
    actually occurred, not batched), one tool message per tool_result --
    the shape `TaskAdherenceEvaluator`/`ToolCallAccuracyEvaluator` expect.
    """
    messages: list[dict[str, Any]] = []
    for event in events:
        if event.kind == "text" and event.text:
            messages.append(_text_message("assistant", event.text))
        elif event.kind == "tool_call":
            # Flat, per the SDK's actual validator
            # (_conversation_validator._validate_tool_call_content_item):
            # name/arguments/tool_call_id live directly on the content item,
            # not nested under a "tool_call"/"function" sub-object the way
            # the evaluators' own docstring examples show it -- live-caught
            # 2026-08-27 ("Each tool_call content items must contain a
            # 'name' field" against the nested shape).
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "name": event.name,
                            "arguments": _parsed_arguments(event.arguments),
                            "tool_call_id": event.call_id,
                        }
                    ],
                }
            )
        elif event.kind == "tool_result":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": event.call_id,
                    "content": [{"type": "tool_result", "tool_result": event.text or ""}],
                }
            )
    return messages


def _tool_definitions(stage: Stage, result: StageResult) -> list[dict[str, Any]]:
    """Schemas for every tool this stage expected or actually called.

    The union, not just `expects_tools`: a stage that called something
    unexpected should still be checkable against that tool's real schema.
    """
    from ds_crew.service.registry import TOOL_CLASSES, args_schema_of, description_of, tool_name_of

    wanted = set(stage.expects_tools) | set(result.tool_calls)
    by_name = {tool_name_of(cls): cls for cls in TOOL_CLASSES}
    return [
        {
            "name": name,
            "description": description_of(by_name[name]),
            "parameters": args_schema_of(by_name[name]).model_json_schema(),
        }
        for name in sorted(wanted)
        if name in by_name
    ]


@dataclass
class ConversationForEval:
    stage_key: str
    query: list[dict[str, Any]]
    response: list[dict[str, Any]]
    tool_definitions: list[dict[str, Any]]
    context: str | None = None


def build_conversation(stage: Stage, state: PipelineState) -> ConversationForEval | None:
    """Assemble one stage's data for TaskAdherence/ToolCallAccuracy (and,
    for `explanation`, Groundedness's `context`). None if the stage never
    ran in this state.
    """
    result = state.results.get(stage.key)
    if result is None:
        return None

    query = [_text_message("user", build_prompt(stage, state))]
    response = _response_messages(result.events) or [_text_message("assistant", result.text)]
    tool_definitions = _tool_definitions(stage, result)

    context = None
    if stage.key == "explanation":
        context = result.tool_results.get("explain_models") or None

    return ConversationForEval(
        stage_key=stage.key,
        query=query,
        response=response,
        tool_definitions=tool_definitions,
        context=context,
    )


def _narration_text(response: list[dict[str, Any]]) -> str:
    """Plain text an assistant actually said, from a `response` message list.

    Skips tool_call-content assistant messages -- naively grabbing
    `content[0]` off every assistant message breaks the moment a stage's
    events include both narration and a tool call, which is the normal case
    (an explainer both narrates and calls `explain_models`).
    """
    return "\n\n".join(
        block["text"]
        for m in response
        if m["role"] == "assistant"
        for block in m["content"]
        if block["type"] == "text"
    )


def stages_with_tool_calls(state: PipelineState) -> list[Stage]:
    return [
        STAGES_BY_KEY[key]
        for key, result in state.results.items()
        if result.succeeded_tools() and key in STAGES_BY_KEY
    ]


@dataclass
class EvaluationSummary:
    evaluation_name: str
    output_path: Path
    studio_url: str | None
    rows: int


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_evaluation(
    state: PipelineState,
    stages: list[Stage],
    *,
    output_dir: Path | None = None,
    push_to_foundry: bool = True,
) -> list[EvaluationSummary]:
    """Run Groundedness (on `explanation`, if present) and TaskAdherence +
    ToolCallAccuracy (on every stage in `stages`) as three separate
    `evaluate()` calls -- each evaluator needs a differently-shaped data
    file (Groundedness's `context` doesn't apply to every stage), so one
    shared call/data file across all three isn't workable.

    Real, billed LLM-judge calls -- one per (stage, evaluator) row. Never
    called from a live pipeline run; only from `ds-crew-maf --evaluate`.
    """
    from azure.ai.evaluation import (
        GroundednessEvaluator,
        TaskAdherenceEvaluator,
        ToolCallAccuracyEvaluator,
        evaluate,
    )

    evaluator_kwargs = _evaluator_kwargs()
    out_dir = output_dir or Path("runs") / state.run_id / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    azure_ai_project = settings.AZURE_FOUNDRY_PROJECT_ENDPOINT if push_to_foundry else None

    conversations = [c for c in (build_conversation(s, state) for s in stages) if c is not None]
    summaries: list[EvaluationSummary] = []

    explanation = next((c for c in conversations if c.stage_key == "explanation"), None)
    if explanation is not None and explanation.context:
        rows = [{"response": _narration_text(explanation.response), "context": explanation.context}]
        path = out_dir / "groundedness.jsonl"
        _write_jsonl(rows, path)
        result = evaluate(
            data=str(path),
            evaluators={"groundedness": GroundednessEvaluator(**evaluator_kwargs)},
            evaluation_name=f"ds-crew-groundedness-{state.run_id}",
            azure_ai_project=azure_ai_project,
            output_path=str(out_dir / "groundedness_results.json"),
        )
        summaries.append(
            EvaluationSummary(
                evaluation_name=f"ds-crew-groundedness-{state.run_id}",
                output_path=out_dir / "groundedness_results.json",
                studio_url=result.get("studio_url"),
                rows=1,
            )
        )

    if conversations:
        rows = [
            {
                "stage": c.stage_key,
                "query": c.query,
                "response": c.response,
                "tool_definitions": c.tool_definitions,
            }
            for c in conversations
        ]
        path = out_dir / "tool_evaluators.jsonl"
        _write_jsonl(rows, path)
        result = evaluate(
            data=str(path),
            evaluators={
                "task_adherence": TaskAdherenceEvaluator(**evaluator_kwargs),
                "tool_call_accuracy": ToolCallAccuracyEvaluator(**evaluator_kwargs),
            },
            evaluation_name=f"ds-crew-tool-evaluators-{state.run_id}",
            azure_ai_project=azure_ai_project,
            output_path=str(out_dir / "tool_evaluators_results.json"),
        )
        summaries.append(
            EvaluationSummary(
                evaluation_name=f"ds-crew-tool-evaluators-{state.run_id}",
                output_path=out_dir / "tool_evaluators_results.json",
                studio_url=result.get("studio_url"),
                rows=len(rows),
            )
        )

    return summaries
