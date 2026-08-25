from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ds_crew import settings
from ds_crew.service.app import create_app
from ds_crew.service.registry import TOOL_CLASSES, tool_name_of

API_KEY = "test-service-key"


@pytest.fixture
def client(monkeypatch):
    """A fresh app per test.

    `create_app` mints run tokens into an instance-local registry, while the
    DataStore it reads is a process singleton that conftest's `_clean_data_store`
    resets between tests. Sharing one app would leave tokens pointing at runs
    whose state had already been cleared.
    """
    monkeypatch.setattr(settings, "SERVICE_API_KEY", API_KEY)
    return TestClient(create_app())


def _create_run(client, classification_df, **overrides) -> dict:
    body = {
        "csv_text": classification_df.to_csv(index=False),
        "target": "target",
        **overrides,
    }
    response = client.post("/runs", json=body, headers={"X-API-Key": API_KEY})
    assert response.status_code == 200, response.text
    return response.json()


# ----------------------------------------------------------------------
# Surface
# ----------------------------------------------------------------------


def test_healthz_lists_every_registered_tool(client):
    payload = client.get("/healthz").json()
    assert payload["status"] == "ok"
    assert set(payload["tools"]) == {tool_name_of(c) for c in TOOL_CLASSES}


def test_openapi_exposes_each_tools_own_argument_schema(client):
    spec = client.get("/openapi.json").json()
    # The whole point of generating routes from the registry: an agent reading
    # this document gets each tool's real Pydantic argument schema, not an
    # untyped blob. A regression here would leave a Foundry agent unable to call
    # the tools correctly while every other test still passed.
    for tool_cls in TOOL_CLASSES:
        path = f"/runs/{{run_id}}/tools/{tool_name_of(tool_cls)}"
        assert path in spec["paths"]
        schema = spec["paths"][path]["post"]["requestBody"]["content"]["application/json"][
            "schema"
        ]
        assert schema["$ref"].endswith(f"/{tool_cls.model_fields['args_schema'].default.__name__}")


# ----------------------------------------------------------------------
# Authorization
# ----------------------------------------------------------------------


def test_create_run_requires_api_key(client, classification_df):
    response = client.post(
        "/runs", json={"csv_text": classification_df.to_csv(index=False), "target": "target"}
    )
    assert response.status_code == 422  # missing required header


def test_create_run_rejects_wrong_api_key(client, classification_df):
    response = client.post(
        "/runs",
        json={"csv_text": classification_df.to_csv(index=False), "target": "target"},
        headers={"X-API-Key": "not-the-key"},
    )
    assert response.status_code == 401


def test_service_refuses_to_create_runs_when_no_key_is_configured(monkeypatch, classification_df):
    # An unset SERVICE_API_KEY must not degrade to an open endpoint: these tools
    # mutate datasets and train models.
    monkeypatch.setattr(settings, "SERVICE_API_KEY", None)
    unconfigured = TestClient(create_app())
    response = unconfigured.post(
        "/runs",
        json={"csv_text": classification_df.to_csv(index=False), "target": "target"},
        headers={"X-API-Key": "anything"},
    )
    assert response.status_code == 503


def test_tool_call_requires_the_run_token(client, classification_df):
    run = _create_run(client, classification_df)
    response = client.post(f"/runs/{run['run_id']}/tools/eda_summary", json={})
    assert response.status_code == 422  # missing required header


def test_run_token_cannot_address_a_different_run(client, classification_df):
    """The property that makes the in-process design defensible, preserved over HTTP.

    In-process, run_id is bound into the tool constructor and never LLM-visible,
    so an agent cannot reach another run's data. Over HTTP run_id is part of the
    request, so a shared key alone would be strictly weaker. Holding run A's
    token must not grant access to run B even when B's id is known.
    """
    run_a = _create_run(client, classification_df)
    run_b = _create_run(client, classification_df)
    assert run_a["run_id"] != run_b["run_id"]

    response = client.post(
        f"/runs/{run_b['run_id']}/tools/eda_summary",
        json={},
        headers={"X-Run-Token": run_a["run_token"]},
    )
    # 404, not 403: the status code must not confirm that run B exists.
    assert response.status_code == 404


def test_unknown_run_id_is_indistinguishable_from_an_unauthorized_one(client, classification_df):
    run = _create_run(client, classification_df)
    forged = client.post(
        "/runs/does-not-exist/tools/eda_summary",
        json={},
        headers={"X-Run-Token": run["run_token"]},
    )
    assert forged.status_code == 404


# ----------------------------------------------------------------------
# Run lifecycle
# ----------------------------------------------------------------------


def test_create_run_infers_task_type_and_seeds_metric(client, classification_df):
    run = _create_run(client, classification_df)
    assert run["task_type"] == "classification"
    assert run["metric_name"]
    assert run["n_rows"] == len(classification_df)
    assert run["n_cols"] == len(classification_df.columns)
    assert run["run_token"]


def test_create_run_rejects_a_target_not_in_the_data(client, classification_df):
    response = client.post(
        "/runs",
        json={"csv_text": classification_df.to_csv(index=False), "target": "nope"},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_create_run_rejects_a_metric_the_task_type_disallows(client, regression_df):
    response = client.post(
        "/runs",
        json={
            "csv_text": regression_df.to_csv(index=False),
            "target": "target",
            "task": "regression",
            "metric": "f1",
        },
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 400


def test_run_status_reports_stage_flags(client, classification_df):
    run = _create_run(client, classification_df)
    status = client.get(
        f"/runs/{run['run_id']}", headers={"X-Run-Token": run["run_token"]}
    ).json()
    assert status["task_type"] == "classification"
    assert status["stages_applied"]["cleaning"] is False
    assert status["stages_applied"]["finalize"] is False


def test_deleting_a_run_revokes_its_token(client, classification_df):
    run = _create_run(client, classification_df)
    headers = {"X-Run-Token": run["run_token"]}
    assert client.delete(f"/runs/{run['run_id']}", headers=headers).status_code == 204
    assert client.get(f"/runs/{run['run_id']}", headers=headers).status_code == 404


# ----------------------------------------------------------------------
# Tool invocation
# ----------------------------------------------------------------------


def test_eda_summary_returns_a_parsed_profile(client, classification_df):
    run = _create_run(client, classification_df)
    response = client.post(
        f"/runs/{run['run_id']}/tools/eda_summary",
        json={},
        headers={"X-Run-Token": run["run_token"]},
    )
    assert response.status_code == 200
    report = response.json()
    # Real JSON, not a quoted string blob.
    assert report["run_id"] == run["run_id"]
    assert report["target"] == "target"
    assert report["n_rows"] == len(classification_df)
    assert any(c["name"] == "constant_col" and c["is_constant"] for c in report["columns"])


def test_body_run_id_must_match_the_path(client, classification_df):
    run = _create_run(client, classification_df)
    response = client.post(
        f"/runs/{run['run_id']}/tools/set_evaluation_metric",
        json={"run_id": "some-other-run", "metric": "f1", "rationale": "x"},
        headers={"X-Run-Token": run["run_token"]},
    )
    assert response.status_code == 400
    assert "does not match" in response.json()["detail"]


def test_out_of_order_tool_call_is_refused_by_the_tool_not_by_http(client, classification_df):
    """Ordering stays a tool-level decision.

    ExplainModelsTool requires evaluation to have run first. That refusal must
    reach the caller as the same {"error": ...} payload an in-process agent
    sees, with a 200, rather than being reshaped into an HTTP error -- the agent
    is expected to read it and correct course.
    """
    run = _create_run(client, classification_df)
    response = client.post(
        f"/runs/{run['run_id']}/tools/explain_models",
        json={},
        headers={"X-Run-Token": run["run_token"]},
    )
    assert response.status_code == 200
    assert "evaluate_models" in response.json()["error"]
