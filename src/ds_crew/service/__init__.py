"""HTTP surface over the deterministic tool layer.

Exists so callers outside this process -- an Azure AI Foundry agent, or a second
DS-Crew replica -- can invoke the same Pydantic-validated tools the in-process
CrewAI orchestrator uses. The orchestrator itself deliberately does NOT go
through this: it keeps calling tools in-process, so adding the service carries
no regression risk for the path that already works.
"""
