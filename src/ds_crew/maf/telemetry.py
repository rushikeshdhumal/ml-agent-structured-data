"""Export agent_framework's OTel traces, plus this project's own failure-rate
metrics, to Application Insights, if configured.

`FoundryAgent`/`Workflow` already emit spans and metrics by default (Microsoft
Agent Framework's instrumentation is on unless explicitly disabled); nothing
in this project has ever configured a destination for them, so today they are
created and immediately dropped. `setup_observability()` is a no-op unless
`settings.APPLICATIONINSIGHTS_CONNECTION_STRING` is set, so that remains true
until someone opts in -- no regression for a run without an App Insights
resource.

`get_meter()` (Phase 11) adds this project's own metrics on top of
agent_framework's default spans: per-stage tool-refusal, human-denial, and
transport-retry counts, which `StageResult` (`ds_crew.foundry.runner`) has
always carried but which `ds_crew.maf.executors` previously only printed to
the console. The old CrewAI implementation had `usage_listener.py` for this;
it was deleted with CrewAI (commit `2ef67a8`) and never replaced until now.
OTel counters are safe no-ops when no `MeterProvider` is configured (the
standard OTel API design), so `executors.py` doesn't need to guard its
`counter.add(...)` calls on whether observability is enabled -- the same
"safe with nothing configured" property `setup_observability()` already
relies on for tracing.

Foundry's own portal (Tracing tab) already shows per-call spans for a
run driven straight from the portal; this wires the same signal for runs
driven by `ds-crew-maf`; so a demo can point at either Application Insights
or Foundry's own trace view for the *operational* record (what actually
happened, per call: latency, tokens, the exact model version served behind a
stage's deployment). MLflow is the *decision* record (the run's leaderboard,
the model `finalize_run` recorded, the human verdict) -- its lifecycle is
tied to a run's HTTP request cycle, see `ds_crew.tools.logging_tools`'s
module docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ds_crew import settings

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter


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


def get_meter() -> Meter:
    """The OTel meter this project's own counters (see `executors.py`) use.

    Safe to call regardless of whether `setup_observability()` found a
    connection string -- with no `MeterProvider` configured, the OTel API
    returns a no-op meter whose counters record nothing and export nowhere,
    rather than raising.
    """
    from opentelemetry import metrics

    return metrics.get_meter("ds_crew.maf")
