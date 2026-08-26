"""Drive DS-Crew's eight Azure AI Foundry agents through a full pipeline run.

Optional package. The tool service does not import it -- this module only ever
calls the service over HTTP/MCP, the same way a Foundry agent does -- and it
needs the `foundry` extra (`uv sync --extra foundry`) plus an Entra login
(`az login`).
"""

from ds_crew.foundry.orchestrator import (
    PreflightError,
    RunReport,
    build_project_client,
    create_run,
    preflight,
    run_pipeline,
)
from ds_crew.foundry.stages import GATED_TOOLS, STAGES, Stage

__all__ = [
    "GATED_TOOLS",
    "STAGES",
    "PreflightError",
    "RunReport",
    "Stage",
    "build_project_client",
    "create_run",
    "preflight",
    "run_pipeline",
]
