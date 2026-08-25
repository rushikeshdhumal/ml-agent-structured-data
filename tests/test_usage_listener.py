from __future__ import annotations

import pytest
from crewai.events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMGuardrailCompletedEvent,
    ToolUsageErrorEvent,
    crewai_event_bus,
)
from crewai.events.types.llm_events import LLMCallType

from ds_crew.usage_listener import PerTaskUsageListener, log_per_task_usage


@pytest.fixture
def listener():
    """A listener registered on CrewAI's process-global event bus.

    The bus is a singleton, so instances registered by earlier tests would also
    observe events emitted here. Each test asserts only on its own instance,
    which stays correct regardless of how many other listeners exist.
    """
    return PerTaskUsageListener()


def _llm_completed(task="eda_task", agent="Senior Data Profiling Analyst", prompt=100,
                   completion=50, cached=0, model="test-model"):
    return LLMCallCompletedEvent(
        task_name=task,
        agent_role=agent,
        model=model,
        call_id="c1",
        response="ok",
        call_type=LLMCallType.LLM_CALL,
        usage={
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_prompt_tokens": cached,
            "total_tokens": prompt + completion,
        },
    )


def _emit(event):
    """Emit and wait for handlers to finish.

    `emit` dispatches sync handlers on a ThreadPoolExecutor and returns a Future
    rather than running them inline, so asserting straight after an emit races
    the handler and sees an empty bucket. Blocking on the future is what the
    bus's own docstring prescribes for sync callers.
    """
    future = crewai_event_bus.emit(source=None, event=event)
    if future is not None:
        future.result(timeout=10)


# ----------------------------------------------------------------------
# Attribution
# ----------------------------------------------------------------------


def test_llm_calls_are_attributed_to_their_task(listener):
    _emit(_llm_completed(task="eda_task", prompt=100, completion=50))
    _emit(_llm_completed(task="eda_task", prompt=200, completion=80))
    _emit(_llm_completed(task="hpo_task", agent="Hyperparameter Optimization Engineer",
                         prompt=10, completion=5))

    by_task = {t.task_name: t for t in listener.snapshot()}
    assert by_task["eda_task"].prompt_tokens == 300
    assert by_task["eda_task"].completion_tokens == 130
    assert by_task["eda_task"].llm_calls == 2
    assert by_task["hpo_task"].prompt_tokens == 10
    assert by_task["hpo_task"].llm_calls == 1


def test_snapshot_preserves_first_seen_order(listener):
    # For a sequential crew, first-seen order is pipeline order, which is what
    # makes the report readable next to tasks.yaml.
    for t in ("eda_task", "propose_cleaning_task", "hpo_task"):
        _emit(_llm_completed(task=t))
    assert [t.task_name for t in listener.snapshot()] == [
        "eda_task",
        "propose_cleaning_task",
        "hpo_task",
    ]


def test_calls_without_a_task_are_bucketed_not_dropped(listener):
    _emit(_llm_completed(task=None, prompt=7, completion=3))
    names = [t.task_name for t in listener.snapshot()]
    assert "(outside task)" in names
    assert listener.snapshot()[0].prompt_tokens == 7


def test_agent_rollup_sums_across_that_agents_tasks(listener):
    role = "Data Cleaning Strategist"
    _emit(_llm_completed(task="propose_cleaning_task", agent=role, prompt=100, completion=40))
    _emit(_llm_completed(task="execute_cleaning_task", agent=role, prompt=60, completion=20))
    _emit(_llm_completed(task="hpo_task", agent="Hyperparameter Optimization Engineer",
                         prompt=5, completion=5))

    agents = listener.by_agent()
    assert agents[role]["prompt_tokens"] == 160
    assert agents[role]["completion_tokens"] == 60
    assert agents[role]["tasks"] == 2
    assert agents["Hyperparameter Optimization Engineer"]["tasks"] == 1


def test_model_and_agent_names_are_recorded(listener):
    _emit(_llm_completed(task="eda_task", agent="Senior Data Profiling Analyst", model="gpt-5"))
    t = listener.snapshot()[0]
    assert t.models == {"gpt-5"}
    assert t.agent_roles == {"Senior Data Profiling Analyst"}


# ----------------------------------------------------------------------
# Retry accounting
# ----------------------------------------------------------------------


def test_failed_guardrails_are_counted_and_passing_ones_are_not(listener):
    """Retry counts are half the model-selection picture: a cheap model that
    retries repeatedly can cost more than an expensive one that succeeds once.
    A guardrail that passes costs nothing extra, so it must not inflate this.
    """
    _emit(LLMGuardrailCompletedEvent(task_name="propose_cleaning_task", success=False,
                                     retry_count=1, result=None))
    _emit(LLMGuardrailCompletedEvent(task_name="propose_cleaning_task", success=False,
                                     retry_count=2, result=None))
    _emit(LLMGuardrailCompletedEvent(task_name="propose_cleaning_task", success=True,
                                     retry_count=3, result=None))
    by_task = {t.task_name: t for t in listener.snapshot()}
    assert by_task["propose_cleaning_task"].guardrail_failures == 2


def test_llm_failures_and_tool_errors_are_counted(listener):
    _emit(LLMCallFailedEvent(task_name="eda_task", call_id="x", error="boom"))
    _emit(ToolUsageErrorEvent(task_name="eda_task", tool_name="eda_summary",
                              tool_args={}, run_attempts=1, error="bad"))
    by_task = {t.task_name: t for t in listener.snapshot()}
    assert by_task["eda_task"].llm_failures == 1
    assert by_task["eda_task"].tool_errors == 1


# ----------------------------------------------------------------------
# Robustness -- instrumentation must never break the run it measures
# ----------------------------------------------------------------------


def test_malformed_usage_payload_does_not_raise(listener):
    """Usage dicts come from provider SDKs. A None or non-numeric field must
    degrade to zero rather than take down a run mid-pipeline.
    """
    _emit(LLMCallCompletedEvent(task_name="eda_task", call_id="c", response="r",
                                call_type=LLMCallType.LLM_CALL,
                                usage={"prompt_tokens": None, "completion_tokens": "abc"}))
    t = listener.snapshot()[0]
    assert t.prompt_tokens == 0
    assert t.completion_tokens == 0
    assert t.llm_calls == 1


def test_missing_usage_entirely_does_not_raise(listener):
    _emit(LLMCallCompletedEvent(task_name="eda_task", call_id="c", response="r",
                                call_type=LLMCallType.LLM_CALL, usage=None))
    assert listener.snapshot()[0].llm_calls == 1


def test_logging_with_no_mlflow_run_id_is_a_no_op(listener):
    # Mirrors every helper in logging_tools: a falsy run_id means "not tracking",
    # not "crash".
    _emit(_llm_completed())
    log_per_task_usage(None, listener)
    log_per_task_usage("", listener)


def test_report_shape_is_stable(listener):
    _emit(_llm_completed(task="eda_task"))
    report = listener.report()
    assert set(report) == {"by_task", "by_agent"}
    assert report["by_task"][0]["task_name"] == "eda_task"
    assert "prompt_tokens" in report["by_task"][0]
