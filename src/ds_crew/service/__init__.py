"""HTTP surface over the deterministic tool layer.

Exists so callers outside this process -- an Azure AI Foundry agent, chiefly --
can invoke the same Pydantic-validated tools defined in `ds_crew.tools`. This
is the only caller those tools have on this branch: `ds_crew.foundry` never
imports them directly, and only ever reaches them over HTTP (this module) or
MCP (`mcp_app.py`), exactly as a Foundry agent does.
"""
