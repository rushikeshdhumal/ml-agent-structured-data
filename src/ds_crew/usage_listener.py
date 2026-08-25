"""Per-task/per-agent LLM usage accounting, via CrewAI's event bus.

`logging_tools.log_llm_usage` records crew-level totals, which answers "what did
this run cost" but not "which agent spent it". Choosing a model per agent needs
the second question answered, so this listener attributes every LLM call to the
task and agent that made it.

No "current task" tracking is needed: `LLMCallCompletedEvent` already carries
`task_name` and `agent_role`, so each call self-identifies. That matters because
CrewAI executes tool calls on worker threads -- any ambient "which task are we
in" state would be wrong there, in the same way `mlflow.active_run()` is
invisible from those threads (see logging_tools.py).

Alongside tokens this records **retry counts** -- guardrail failures and tool
errors -- because a cheap model that fails a guardrail three times can cost more
than an expensive one that succeeds once. Cost per agent is only half the model
selection picture; reliability per agent is the other half.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from crewai.events import (
    BaseEventListener,
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMGuardrailCompletedEvent,
    ToolUsageErrorEvent,
)

_NO_TASK = "(outside task)"


@dataclass
class TaskUsage:
    """Accumulated usage for one task. Counters only -- no interpretation here."""

    task_name: str
    agent_roles: set[str] = field(default_factory=set)
    models: set[str] = field(default_factory=set)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    guardrail_failures: int = 0
    tool_errors: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "agent_roles": sorted(self.agent_roles),
            "models": sorted(self.models),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "llm_failures": self.llm_failures,
            "guardrail_failures": self.guardrail_failures,
            "tool_errors": self.tool_errors,
        }


def _as_int(value: Any) -> int:
    """Usage dicts come from provider SDKs and occasionally carry None or a
    float. Never let a malformed field take down a run.
    """
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class PerTaskUsageListener(BaseEventListener):
    """Buckets LLM usage and retries by task.

    Registers on CrewAI's process-global `crewai_event_bus`, so one instance per
    run in a one-run-per-process CLI. Instances accumulate independently, so a
    long-lived process that builds several listeners will have each of them see
    every subsequent event -- fine for `ds-crew`, worth knowing in tests.

    Every handler is defensive: instrumentation that raises would take down the
    run it is measuring, which is a strictly worse outcome than losing the
    measurement.
    """

    def __init__(self) -> None:
        # Set state up before super().__init__(), which calls setup_listeners()
        # and would otherwise close over attributes that do not exist yet.
        self._lock = threading.Lock()
        self._by_task: dict[str, TaskUsage] = {}
        self._order: list[str] = []
        super().__init__()

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def _bucket(self, task_name: str | None) -> TaskUsage:
        """Caller must hold the lock."""
        name = task_name or _NO_TASK
        if name not in self._by_task:
            self._by_task[name] = TaskUsage(task_name=name)
            self._order.append(name)
        return self._by_task[name]

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _on_llm_completed(source: Any, event: Any) -> None:
            try:
                usage = event.usage or {}
                with self._lock:
                    b = self._bucket(getattr(event, "task_name", None))
                    if getattr(event, "agent_role", None):
                        b.agent_roles.add(event.agent_role)
                    if getattr(event, "model", None):
                        b.models.add(event.model)
                    b.prompt_tokens += _as_int(usage.get("prompt_tokens"))
                    b.completion_tokens += _as_int(usage.get("completion_tokens"))
                    b.cached_prompt_tokens += _as_int(usage.get("cached_prompt_tokens"))
                    b.total_tokens += _as_int(usage.get("total_tokens"))
                    b.llm_calls += 1
            except Exception:
                pass

        @crewai_event_bus.on(LLMCallFailedEvent)
        def _on_llm_failed(source: Any, event: Any) -> None:
            try:
                with self._lock:
                    self._bucket(getattr(event, "task_name", None)).llm_failures += 1
            except Exception:
                pass

        @crewai_event_bus.on(LLMGuardrailCompletedEvent)
        def _on_guardrail(source: Any, event: Any) -> None:
            # Only failures are counted. A guardrail that passes costs nothing
            # extra; a failure forces the whole task prompt to be resent.
            try:
                if getattr(event, "success", True):
                    return
                with self._lock:
                    self._bucket(getattr(event, "task_name", None)).guardrail_failures += 1
            except Exception:
                pass

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def _on_tool_error(source: Any, event: Any) -> None:
            try:
                with self._lock:
                    self._bucket(getattr(event, "task_name", None)).tool_errors += 1
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def snapshot(self) -> list[TaskUsage]:
        """Per-task usage in first-seen order, which for a sequential crew is
        pipeline order.
        """
        with self._lock:
            return [self._by_task[n] for n in self._order]

    def by_agent(self) -> dict[str, dict[str, int]]:
        """Roll tasks up to agents, which is the granularity model selection
        actually operates at.

        A task's tokens are attributed whole to each agent that made a call
        during it. In this pipeline every task has exactly one agent, so the
        rollup is exact; were a task ever shared, its tokens would be
        double-counted and the totals would need revisiting.
        """
        agents: dict[str, dict[str, int]] = {}
        for t in self.snapshot():
            for role in t.agent_roles or {"(unknown)"}:
                a = agents.setdefault(
                    role,
                    {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cached_prompt_tokens": 0,
                        "total_tokens": 0,
                        "llm_calls": 0,
                        "llm_failures": 0,
                        "guardrail_failures": 0,
                        "tool_errors": 0,
                        "tasks": 0,
                    },
                )
                a["prompt_tokens"] += t.prompt_tokens
                a["completion_tokens"] += t.completion_tokens
                a["cached_prompt_tokens"] += t.cached_prompt_tokens
                a["total_tokens"] += t.total_tokens
                a["llm_calls"] += t.llm_calls
                a["llm_failures"] += t.llm_failures
                a["guardrail_failures"] += t.guardrail_failures
                a["tool_errors"] += t.tool_errors
                a["tasks"] += 1
        return agents

    def report(self) -> dict[str, Any]:
        return {
            "by_task": [t.as_dict() for t in self.snapshot()],
            "by_agent": self.by_agent(),
        }


def log_per_task_usage(run_id: str | None, listener: PerTaskUsageListener) -> None:
    """Write the breakdown to MLflow: metrics for the headline numbers, plus the
    full table as a JSON artifact.

    Called from the same `finally` as `log_llm_usage`, so a run that burned
    tokens and then crashed still reports where they went. Never raises, for the
    same reason the handlers do not.

    One caveat inherent to the bus: `emit` dispatches sync handlers on a
    ThreadPoolExecutor rather than inline, so an event emitted moments before
    kickoff returns could still be in flight when this runs. In practice the
    gap is covered by `log_llm_usage` doing MLflow I/O first, but the honest
    check is arithmetic: the per-task token sums should match the crew-level
    totals `log_llm_usage` records. A shortfall means events were missed, not
    that the pipeline used fewer tokens.
    """
    try:
        from ds_crew.tools.logging_tools import log_json_artifact, log_metrics

        if not run_id:
            return
        report = listener.report()
        log_json_artifact(run_id, report, "usage/per_task_usage.json")

        metrics: dict[str, float] = {}
        for task in listener.snapshot():
            # Slashes group these in the MLflow UI. Task names come from
            # tasks.yaml keys, which are already metric-safe identifiers.
            metrics[f"usage/{task.task_name}/prompt_tokens"] = task.prompt_tokens
            metrics[f"usage/{task.task_name}/completion_tokens"] = task.completion_tokens
            metrics[f"usage/{task.task_name}/llm_calls"] = task.llm_calls
            if task.guardrail_failures:
                metrics[f"usage/{task.task_name}/guardrail_failures"] = task.guardrail_failures
            if task.tool_errors:
                metrics[f"usage/{task.task_name}/tool_errors"] = task.tool_errors
        if metrics:
            log_metrics(run_id, metrics)
    except Exception:
        pass
