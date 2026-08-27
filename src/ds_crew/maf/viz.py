"""Render the workflow's topology as a Mermaid diagram.

Foundry's in-portal drag-and-drop Workflow canvas retires 2026-12-01; this is
the answer to "can I see the orchestration?" without it -- and unlike a
hand-drawn diagram, it's generated from the same `Workflow` object that
executes, so it can't drift out of sync with the real graph.
"""

from __future__ import annotations

from typing import Any

from agent_framework import WorkflowViz

from ds_crew.maf.workflow import build_workflow


class _UnusedTransport:
    """Satisfies `build_workflow`'s transport parameter for graph construction
    only. Building the graph never calls a transport method -- only running
    the workflow does -- so this exists purely to avoid needing real Azure
    credentials just to draw a diagram."""

    async def start(self, *, stage: Any, prompt: str, conversation: str | None) -> Any:
        raise NotImplementedError("This transport is for workflow-diagram generation only.")

    async def answer(self, *, stage: Any, conversation: str, answers: Any) -> Any:
        raise NotImplementedError("This transport is for workflow-diagram generation only.")


async def _unused_decider(req: Any) -> tuple[bool, str]:
    raise NotImplementedError


async def _unused_verdict_collector(explanation_text: str) -> str:
    raise NotImplementedError


def to_mermaid() -> str:
    workflow = build_workflow(
        transport=_UnusedTransport(),
        decide=_unused_decider,
        collect_verdict=_unused_verdict_collector,
    )
    return WorkflowViz(workflow).to_mermaid()


def write_mermaid(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_mermaid())
