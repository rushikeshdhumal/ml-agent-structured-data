from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ds_crew import settings
from ds_crew.service.app import create_app
from ds_crew.service.mcp_app import _allowed_hosts, build_mcp_server
from ds_crew.service.registry import TOOL_CLASSES, tool_name_of

API_KEY = "test-mcp-key"
_HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}
_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "tests", "version": "1"},
    },
}


@pytest.fixture
def client(monkeypatch):
    """A client with the MCP session manager actually running.

    TestClient must be entered as a context manager: mounting the MCP app does
    not start its session manager, the FastAPI lifespan does, and without it the
    endpoint accepts connections and then hangs.
    """
    monkeypatch.setattr(settings, "SERVICE_API_KEY", API_KEY)
    with TestClient(create_app()) as c:
        yield c


def _rpc(client, method: str, params: dict | None = None, req_id: int = 2):
    """One JSON-RPC round trip, unwrapping the SSE framing."""
    client.post("/mcp", json=_INIT, headers=_HEADERS)
    client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=_HEADERS
    )
    body = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
    response = client.post("/mcp", json=body, headers=_HEADERS)
    assert response.status_code == 200, response.text
    data = [ln for ln in response.text.splitlines() if ln.startswith("data:")]
    assert data, f"no SSE data frame in: {response.text[:300]}"
    return json.loads(data[0][len("data:") :])


# ----------------------------------------------------------------------
# Tool surface
# ----------------------------------------------------------------------


def test_every_registry_tool_is_exposed(client):
    tools = _rpc(client, "tools/list")["result"]["tools"]
    assert {t["name"] for t in tools} == {tool_name_of(c) for c in TOOL_CLASSES}


def test_every_tool_takes_run_id(client):
    """Foundry's mcp tool sends only static headers, so a per-run credential is
    unrepresentable and run_id has to travel as an argument on every call.
    """
    for tool in _rpc(client, "tools/list")["result"]["tools"]:
        assert "run_id" in tool["inputSchema"]["properties"], tool["name"]
        assert "run_id" in tool["inputSchema"]["required"], tool["name"]


def test_schemas_are_flat_not_wrapped_in_one_object(client):
    """A single Pydantic parameter would nest every field under one property.

    Flat schemas are materially easier for a model to fill in, which is why the
    tool callables are built with dynamic explicit signatures. `$defs` may still
    appear where a model genuinely nests (cleaning actions, column plans).
    """
    by_name = {t["name"]: t for t in _rpc(client, "tools/list")["result"]["tools"]}
    eda = by_name["eda_summary"]["inputSchema"]["properties"]
    assert set(eda) == {"run_id", "include_correlations"}

    cleaning = by_name["apply_cleaning_plan"]["inputSchema"]["properties"]
    assert {"run_id", "actions", "drop_duplicate_rows", "columns_to_drop"} <= set(cleaning)


def test_nested_item_schemas_are_inlined_not_left_as_refs(client):
    """A live Foundry agent invented the nested shape when it was behind a $ref.

    It produced every top-level field of the feature plan correctly and every
    nested field wrong -- `name` for `column`, `encoding: "numeric"`, plus
    `transformer` and `handle_unknown`, which do not exist. The failure boundary
    was exactly the reference, so the item shape has to be stated inline.
    """
    by_name = {t["name"]: t for t in _rpc(client, "tools/list")["result"]["tools"]}

    for tool_name, array_field, expected_fields in (
        ("apply_feature_plan", "column_plans", {"column", "encoding", "scaling"}),
        ("apply_cleaning_plan", "actions", {"column", "missing_strategy", "outlier_strategy"}),
    ):
        schema = by_name[tool_name]["inputSchema"]
        assert "$defs" not in schema, tool_name
        assert "$ref" not in json.dumps(schema), tool_name

        items = schema["properties"][array_field]["items"]
        assert expected_fields <= set(items["properties"]), tool_name


def test_inlined_enums_are_visible_on_nested_fields(client):
    """The allowed values are the half the agent got wrong; they must be inline."""
    by_name = {t["name"]: t for t in _rpc(client, "tools/list")["result"]["tools"]}
    items = by_name["apply_feature_plan"]["inputSchema"]["properties"]["column_plans"]["items"]
    assert set(items["properties"]["encoding"]["enum"]) == {
        "onehot",
        "ordinal",
        "target_mean",
        "frequency",
        "none",
    }
    assert set(items["properties"]["scaling"]["enum"]) == {
        "standard",
        "minmax",
        "robust",
        "none",
    }


def test_inlining_does_not_weaken_validation(client, classification_run):
    """Inlining rewrites what is advertised, never what is enforced."""
    result = _rpc(
        client,
        "tools/call",
        {
            "name": "apply_feature_plan",
            "arguments": {
                "run_id": classification_run.run_id,
                # The exact shape the live agent sent.
                "column_plans": [{"name": "num_a", "encoding": "numeric"}],
            },
        },
    )
    assert "error" in json.dumps(result).lower()


def test_tool_descriptions_come_from_the_tool_classes(client):
    by_name = {t["name"]: t for t in _rpc(client, "tools/list")["result"]["tools"]}
    expected = TOOL_CLASSES[0].model_fields["description"].default
    assert by_name[tool_name_of(TOOL_CLASSES[0])]["description"] == expected


# ----------------------------------------------------------------------
# Invocation
# ----------------------------------------------------------------------


def test_calling_a_tool_runs_it_and_returns_its_output(client, classification_run):
    result = _rpc(
        client,
        "tools/call",
        {"name": "eda_summary", "arguments": {"run_id": classification_run.run_id}},
    )
    text = result["result"]["content"][0]["text"]
    report = json.loads(text)
    assert report["run_id"] == classification_run.run_id
    assert report["target"] == "target"
    assert any(c["name"] == "constant_col" and c["is_constant"] for c in report["columns"])


def test_unknown_run_id_returns_an_error_payload_not_a_protocol_error(client):
    """Tools signal refusal in-band and the agent is expected to correct course.

    Raising instead would surface to the agent as a transport failure, which
    reads like an outage rather than a decision it can act on.
    """
    result = _rpc(
        client, "tools/call", {"name": "eda_summary", "arguments": {"run_id": "no-such-run"}}
    )
    text = result["result"]["content"][0]["text"]
    assert "error" in text.lower()
    assert "no-such-run" in text


def test_out_of_order_call_is_refused_by_the_tool(client, classification_run):
    result = _rpc(
        client,
        "tools/call",
        {"name": "explain_models", "arguments": {"run_id": classification_run.run_id}},
    )
    text = result["result"]["content"][0]["text"]
    assert "evaluate_models" in text


# ----------------------------------------------------------------------
# Authorization and transport
# ----------------------------------------------------------------------


def test_mcp_requires_the_api_key(client):
    unauthenticated = {k: v for k, v in _HEADERS.items() if k != "X-API-Key"}
    assert client.post("/mcp", json=_INIT, headers=unauthenticated).status_code == 401
    assert client.post("/mcp", json=_INIT, headers={**_HEADERS, "X-API-Key": "no"}).status_code == 401


def test_unknown_paths_still_404_rather_than_401(client):
    """The MCP app is mounted at "/" to avoid a redirect, so it sees every
    unmatched path. Guarding only /mcp keeps a mistyped URL an honest 404.
    """
    assert client.get("/definitely-not-a-route").status_code == 404


def test_rest_surface_still_works_alongside_mcp(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/openapi-agent.json").status_code == 200


def test_allowed_hosts_includes_the_public_url_host(monkeypatch):
    """DNS-rebinding protection is on by default with an empty allowlist, which
    answers Foundry with HTTP 421. Deriving the allowlist from
    SERVICE_PUBLIC_URL keeps the protection while staying correct as the tunnel
    hostname rotates.
    """
    monkeypatch.setattr(settings, "SERVICE_PUBLIC_URL", "https://abc-8000.use2.devtunnels.ms/")
    hosts = _allowed_hosts()
    assert "abc-8000.use2.devtunnels.ms" in hosts
    assert "abc-8000.use2.devtunnels.ms:443" in hosts
    assert "localhost" in hosts


def test_allowed_hosts_without_public_url_is_local_only(monkeypatch):
    monkeypatch.setattr(settings, "SERVICE_PUBLIC_URL", None)
    hosts = _allowed_hosts()
    assert "localhost" in hosts
    assert not any("devtunnels" in h for h in hosts)


def test_server_builds_without_a_public_url(monkeypatch):
    monkeypatch.setattr(settings, "SERVICE_PUBLIC_URL", None)
    assert build_mcp_server() is not None
