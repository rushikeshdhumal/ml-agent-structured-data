"""`setup_observability()` is a no-op unless an App Insights connection string
is configured, and only opts into sensitive-data capture when explicitly told
to -- see `ds_crew.maf.telemetry`'s module docstring for why either half
matters (a bare `ds-crew-maf` invocation with no Azure Monitor resource must
not fail, and prompt/response content must not leak to a shared telemetry
backend by accident).
"""

from __future__ import annotations

import pytest

from ds_crew.maf.telemetry import setup_observability


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch):
    monkeypatch.setattr("ds_crew.settings.APPLICATIONINSIGHTS_CONNECTION_STRING", None)
    monkeypatch.setattr("ds_crew.settings.ENABLE_SENSITIVE_TELEMETRY", False)


def test_does_nothing_without_a_connection_string(monkeypatch):
    configure = pytest.importorskip("azure.monitor.opentelemetry")
    called = []
    monkeypatch.setattr(configure, "configure_azure_monitor", lambda **kw: called.append(kw))

    setup_observability()

    assert called == []


def test_wires_azure_monitor_when_a_connection_string_is_set(monkeypatch):
    monkeypatch.setattr(
        "ds_crew.settings.APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=abc123"
    )
    azure_monitor = pytest.importorskip("azure.monitor.opentelemetry")
    calls = []
    monkeypatch.setattr(azure_monitor, "configure_azure_monitor", lambda **kw: calls.append(kw))

    setup_observability()

    assert calls == [{"connection_string": "InstrumentationKey=abc123"}]


def test_sensitive_telemetry_stays_off_unless_explicitly_enabled(monkeypatch):
    monkeypatch.setattr(
        "ds_crew.settings.APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=abc123"
    )
    azure_monitor = pytest.importorskip("azure.monitor.opentelemetry")
    monkeypatch.setattr(azure_monitor, "configure_azure_monitor", lambda **kw: None)
    observability = pytest.importorskip("agent_framework.observability")
    calls = []
    monkeypatch.setattr(observability, "enable_sensitive_telemetry", lambda: calls.append(True))

    setup_observability()
    assert calls == []

    monkeypatch.setattr("ds_crew.settings.ENABLE_SENSITIVE_TELEMETRY", True)
    setup_observability()
    assert calls == [True]
