"""Drive DS-Crew's eight Azure AI Foundry agents through a full pipeline run,
via Microsoft Agent Framework instead of a hand-written orchestration loop.

Optional package: needs the `maf` extra (`uv sync --extra maf`) plus an Entra
login (`az login`). Reuses `ds_crew.foundry.stages` (the pipeline definition)
and `ds_crew.foundry.runner` (`is_transport_error`) unchanged; everything else
in `ds_crew.foundry` that this replaces (`orchestrator.py`, its CLI) does not
exist on this branch. The tool service does not import this package -- it only
ever reaches it over HTTP/MCP, the same way a Foundry agent does.
"""

from __future__ import annotations

from ds_crew.foundry.stages import GATED_TOOLS, STAGES, Stage
from ds_crew.maf.state import PipelineState
from ds_crew.maf.transport_foundry import FoundryTransport
from ds_crew.maf.workflow import build_workflow

__all__ = [
    "GATED_TOOLS",
    "STAGES",
    "Stage",
    "PipelineState",
    "FoundryTransport",
    "build_workflow",
]
