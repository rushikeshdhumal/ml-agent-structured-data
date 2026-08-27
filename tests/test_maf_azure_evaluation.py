"""Unit tests for `ds_crew.maf.azure_evaluation`'s data assembly.

No live API calls here -- `evaluate()`/the three evaluator classes make
real, billed LLM-judge calls and are verified live, once, deliberately (see
the module docstring). This file only exercises `build_conversation()` and
`stages_with_tool_calls()`, which are pure functions of a `PipelineState`.
"""

from __future__ import annotations

from ds_crew.foundry.runner import StageResult, ToolEvent
from ds_crew.foundry.stages import STAGES_BY_KEY
from ds_crew.maf.azure_evaluation import (
    _narration_text,
    build_conversation,
    stages_with_tool_calls,
)
from ds_crew.maf.state import PipelineState


def test_returns_none_for_a_stage_that_never_ran():
    state = PipelineState(run_id="r1")
    assert build_conversation(STAGES_BY_KEY["eda"], state) is None


def test_query_wraps_the_stage_prompt_as_a_user_message():
    state = PipelineState(run_id="r1")
    state = state.with_result(
        "eda", StageResult(text="ok", response_id="x", ok_tools=["eda_summary"]), "x"
    )
    conv = build_conversation(STAGES_BY_KEY["eda"], state)
    assert conv.query == [
        {"role": "user", "content": [{"type": "text", "text": "Profile the dataset for run r1."}]}
    ]


def test_response_reconstructs_events_in_order_with_parsed_arguments():
    result = StageResult(
        text="ignored when events are present",
        response_id="x",
        ok_tools=["eda_summary"],
        events=[
            ToolEvent(kind="text", text="Let me profile this dataset."),
            ToolEvent(
                kind="tool_call",
                call_id="call_1",
                name="eda_summary",
                arguments='{"run_id": "r1"}',
            ),
            ToolEvent(kind="tool_result", call_id="call_1", name="eda_summary", text='{"n_rows": 200}'),
        ],
    )
    state = PipelineState(run_id="r1").with_result("eda", result, "x")

    conv = build_conversation(STAGES_BY_KEY["eda"], state)

    assert conv.response == [
        {"role": "assistant", "content": [{"type": "text", "text": "Let me profile this dataset."}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "name": "eda_summary",
                    "arguments": {"run_id": "r1"},
                    "tool_call_id": "call_1",
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "tool_result", "tool_result": '{"n_rows": 200}'}],
        },
    ]


def test_response_falls_back_to_flat_text_when_there_are_no_events():
    result = StageResult(text="narration only, no events captured", response_id="x", ok_tools=["eda_summary"])
    state = PipelineState(run_id="r1").with_result("eda", result, "x")

    conv = build_conversation(STAGES_BY_KEY["eda"], state)

    assert conv.response == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "narration only, no events captured"}],
        }
    ]


def test_multiple_calls_to_the_same_tool_each_keep_their_own_result():
    """The exact case Phase 9's dict-based tool_results (keyed by name) could
    not represent -- a second call to the same tool overwrote the first
    result. events, keyed by call_id, keeps both.
    """
    result = StageResult(
        text="",
        response_id="x",
        ok_tools=["tune_model_hyperparameters", "tune_model_hyperparameters"],
        events=[
            ToolEvent(
                kind="tool_call",
                call_id="call_1",
                name="tune_model_hyperparameters",
                arguments='{"model_names": ["xgboost"]}',
            ),
            ToolEvent(kind="tool_result", call_id="call_1", text='{"xgboost": {}}'),
            ToolEvent(
                kind="tool_call",
                call_id="call_2",
                name="tune_model_hyperparameters",
                arguments='{"model_names": ["lightgbm"]}',
            ),
            ToolEvent(kind="tool_result", call_id="call_2", text='{"lightgbm": {}}'),
        ],
    )
    state = PipelineState(run_id="r1").with_result("hpo", result, "x")

    conv = build_conversation(STAGES_BY_KEY["hpo"], state)

    tool_call_ids = [
        m["content"][0]["tool_call_id"]
        for m in conv.response
        if m["content"][0]["type"] == "tool_call"
    ]
    tool_result_texts = [
        m["content"][0]["tool_result"] for m in conv.response if m["role"] == "tool"
    ]
    assert tool_call_ids == ["call_1", "call_2"]
    assert tool_result_texts == ['{"xgboost": {}}', '{"lightgbm": {}}']


def test_tool_definitions_include_expected_and_actually_called_tools():
    result = StageResult(text="", response_id="x", ok_tools=["eda_summary"], tool_calls=["eda_summary"])
    state = PipelineState(run_id="r1").with_result("eda", result, "x")

    conv = build_conversation(STAGES_BY_KEY["eda"], state)

    names = {d["name"] for d in conv.tool_definitions}
    assert "eda_summary" in names
    for definition in conv.tool_definitions:
        assert "description" in definition
        assert "parameters" in definition


def test_context_is_only_set_for_the_explanation_stage():
    non_explanation = StageResult(text="", response_id="x", ok_tools=["eda_summary"])
    state = PipelineState(run_id="r1").with_result("eda", non_explanation, "x")
    conv = build_conversation(STAGES_BY_KEY["eda"], state)
    assert conv.context is None

    explanation = StageResult(
        text="the explainer's narration",
        response_id="y",
        ok_tools=["explain_models"],
        tool_results={"explain_models": '{"reports": [{"model_name": "xgboost"}]}'},
    )
    state = state.with_result("explanation", explanation, "y")
    conv = build_conversation(STAGES_BY_KEY["explanation"], state)
    assert conv.context == '{"reports": [{"model_name": "xgboost"}]}'


def test_narration_text_skips_tool_call_shaped_assistant_messages():
    """The exact bug caught in review: a stage's response normally mixes
    text and tool_call assistant messages (an explainer both narrates and
    calls explain_models) -- naively reading content[0]["text"] off every
    assistant message crashes on the tool_call ones, which have no "text" key.
    """
    result = StageResult(
        text="",
        response_id="x",
        ok_tools=["explain_models"],
        events=[
            ToolEvent(kind="text", text="Explaining xgboost now."),
            ToolEvent(kind="tool_call", call_id="c1", name="explain_models", arguments="{}"),
            ToolEvent(kind="tool_result", call_id="c1", text="{}"),
            ToolEvent(kind="text", text="xgboost relies most on num_a."),
        ],
    )
    state = PipelineState(run_id="r1").with_result("explanation", result, "x")
    conv = build_conversation(STAGES_BY_KEY["explanation"], state)

    assert _narration_text(conv.response) == "Explaining xgboost now.\n\nxgboost relies most on num_a."


def test_stages_with_tool_calls_filters_to_stages_that_actually_succeeded():
    state = PipelineState(run_id="r1")
    state = state.with_result(
        "eda", StageResult(text="", response_id="x", ok_tools=["eda_summary"]), "x"
    )
    state = state.with_result(
        "cleaning", StageResult(text="", response_id="y", refused_tools=["apply_cleaning_plan"]), "y"
    )

    stages = stages_with_tool_calls(state)

    assert [s.key for s in stages] == ["eda"]
