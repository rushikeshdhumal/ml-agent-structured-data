"""Unit tests for `ds_crew.maf.model_lifecycle`'s cross-referencing logic.

No live Azure calls here -- `fetch_deployments`/`fetch_model_catalog`/
`agent_rai_policies` make real management-plane/Foundry calls and are
verified live, deliberately, once (see the module docstring). This file
only exercises `deployment_health()`, a pure function of already-fetched
(raw, Azure-shaped) JSON.
"""

from __future__ import annotations

from datetime import date

from ds_crew.maf.model_lifecycle import LifecycleReport, deployment_health


def _deployment(name: str, model_name: str, model_version: str = "1") -> dict:
    return {"name": name, "properties": {"model": {"name": model_name, "version": model_version}}}


def _catalog_entry(
    model_name: str,
    model_version: str = "1",
    *,
    lifecycle_status: str = "GenerallyAvailable",
    inference_deprecation: str | None = None,
    tool_calling: bool = True,
) -> dict:
    return {
        "model": {
            "name": model_name,
            "version": model_version,
            "lifecycleStatus": lifecycle_status,
            "deprecation": {"inference": inference_deprecation},
            "capabilities": {"toolCalling": "true" if tool_calling else "false"},
        }
    }


def test_active_model_reports_healthy_with_no_deprecation():
    rows = deployment_health(
        [_deployment("ds-standard", "gpt-5-mini")],
        [_catalog_entry("gpt-5-mini")],
        today=date(2026, 8, 27),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.lifecycle_status == "GenerallyAvailable"
    assert row.inference_deprecation is None
    assert row.days_until_deprecation is None
    assert row.migration_candidates == []


def test_model_within_deprecation_window_gets_migration_candidates():
    rows = deployment_health(
        [_deployment("ds-standard", "gpt-5-mini")],
        [
            _catalog_entry("gpt-5-mini", inference_deprecation="2026-10-01"),
            _catalog_entry("gpt-5.4-mini", inference_deprecation=None),
        ],
        today=date(2026, 8, 27),
    )
    row = rows[0]
    assert row.days_until_deprecation == 35
    assert row.migration_candidates == ["gpt-5.4-mini"]


def test_model_already_past_deprecation_is_flagged_at_report_level():
    rows = deployment_health(
        [_deployment("ds-standard", "gpt-5-mini")],
        [
            _catalog_entry("gpt-5-mini", inference_deprecation="2026-08-20"),
            _catalog_entry("gpt-5.4-mini", inference_deprecation=None),
        ],
        today=date(2026, 8, 27),
    )
    row = rows[0]
    assert row.days_until_deprecation == -7
    assert row.migration_candidates == ["gpt-5.4-mini"]

    report = LifecycleReport(deployments=rows, agents=[])
    assert report.has_past_due_deprecation() is True


def test_deployment_whose_model_is_missing_from_the_catalog_reports_unknown_not_a_crash():
    rows = deployment_health(
        [_deployment("ds-preview", "some-preview-model")],
        [_catalog_entry("gpt-5-mini")],
        today=date(2026, 8, 27),
    )
    row = rows[0]
    assert row.lifecycle_status == "unknown"
    assert row.inference_deprecation is None
    assert row.migration_candidates == []


def test_migration_candidates_exclude_self_deprecated_and_non_tool_calling_models():
    rows = deployment_health(
        [_deployment("ds-standard", "gpt-5-mini")],
        [
            _catalog_entry("gpt-5-mini", inference_deprecation="2026-09-01"),
            _catalog_entry("gpt-5.4-mini"),  # active, tool-calling -- eligible
            _catalog_entry("gpt-5-legacy", lifecycle_status="Deprecated"),  # excluded
            _catalog_entry("gpt-5-vision", tool_calling=False),  # excluded
            _catalog_entry(
                "gpt-5-nano-preview", inference_deprecation="2026-09-05"
            ),  # excluded -- itself within the warning window
        ],
        today=date(2026, 8, 27),
    )
    assert rows[0].migration_candidates == ["gpt-5.4-mini"]


def test_report_with_no_past_due_deployments_is_not_flagged():
    rows = deployment_health(
        [_deployment("ds-standard", "gpt-5-mini")],
        [_catalog_entry("gpt-5-mini")],
        today=date(2026, 8, 27),
    )
    report = LifecycleReport(deployments=rows, agents=[])
    assert report.has_past_due_deprecation() is False
