"""Export agent_framework's OTel traces to Application Insights, if configured.

`FoundryAgent`/`Workflow` already emit spans and metrics by default (Microsoft
Agent Framework's instrumentation is on unless explicitly disabled); nothing
in this project has ever configured a destination for them, so today they are
created and immediately dropped. `setup_observability()` is a no-op unless
`settings.APPLICATIONINSIGHTS_CONNECTION_STRING` is set, so that remains true
until someone opts in -- no regression for a run without an App Insights
resource.

Foundry's own portal (Tracing tab) already shows per-call spans for a
run driven straight from the portal; this wires the same signal for runs
driven by `ds-crew-maf`; so a demo can point at either Application Insights
or Foundry's own trace view for the *operational* record (what actually
happened, per call: latency, tokens, the exact model version served behind a
stage's deployment). MLflow is meant to be the *decision* record (the run's
leaderboard, the model finalize_run recorded, the human verdict) -- but see
`ds_crew.tools.logging_tools`'s module docstring: nothing currently sets
`RunState.mlflow_run_id`, so every MLflow write in this codebase is a live
no-op today. That gap is independent of this module and not closed by it.
"""

from __future__ import annotations

from ds_crew import settings


def setup_observability() -> None:
    """Route agent_framework's OTel output to Application Insights.

    Call once, at process startup, before any agent/workflow runs. Safe to
    call with no connection string configured -- does nothing.
    """
    if not settings.APPLICATIONINSIGHTS_CONNECTION_STRING:
        return

    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor(connection_string=settings.APPLICATIONINSIGHTS_CONNECTION_STRING)

    if settings.ENABLE_SENSITIVE_TELEMETRY:
        from agent_framework.observability import enable_sensitive_telemetry

        enable_sensitive_telemetry()
