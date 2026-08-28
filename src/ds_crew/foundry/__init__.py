"""The pipeline definition Foundry has nowhere else to put.

On this branch (`feat/maf-workflow`), this package holds only the two things
that survive unchanged across an orchestration-framework port: `stages.py`
(the 9-stage pipeline graph -- pure data, zero Azure imports) and `runner.py`
(transport-fault/refusal classification plus the `ToolEvent`/`StageResult`
dataclasses, all reused by `ds_crew.maf`). The orchestrator that used to
live here (`orchestrator.py`, its CLI, and `runner.py`'s own `AgentRunner`)
is gone: driving the pipeline is `ds_crew.maf`'s job now, via Microsoft
Agent Framework. The tool service does not import either package -- it only
ever gets called over HTTP/MCP, the same way a Foundry agent does.
"""

from ds_crew.foundry.stages import GATED_TOOLS, STAGES, Stage

__all__ = [
    "GATED_TOOLS",
    "STAGES",
    "Stage",
]
