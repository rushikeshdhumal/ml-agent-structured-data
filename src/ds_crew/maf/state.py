"""The pipeline's state, carried stage to stage along the workflow's edges.

Replaces `ds_crew.foundry.orchestrator.RunReport`, which the old orchestrator
threaded through a plain Python `for` loop. Kept a fully-serializable
dataclass -- `conversations` holds ids, not live `AgentSession` objects, and
`results` holds `StageResult` (already a plain dataclass) -- so that whatever a
checkpoint captures after a stage completes is honest about what it saved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from ds_crew import settings
from ds_crew.foundry.runner import StageResult


@dataclass
class PipelineState:
    run_id: str
    results: dict[str, StageResult] = field(default_factory=dict)
    # stage key -> transport-specific conversation id (e.g. an AgentSession's
    # session_id). Never a live session object -- see the module docstring.
    conversations: dict[str, str] = field(default_factory=dict)
    verdict: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.results.values())

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results.values())

    @property
    def transport_retries(self) -> int:
        return sum(r.transport_retries for r in self.results.values())

    def cost_usd(self) -> float | None:
        """Only when both rates are configured; otherwise no number at all.

        Reporting $0.00 against unconfigured rates would be a lie dressed as a
        measurement -- ported verbatim from `RunReport.cost_usd`'s reasoning.
        """
        cin, cout = settings.LLM_PRICE_PER_1M_INPUT, settings.LLM_PRICE_PER_1M_OUTPUT
        if cin is None or cout is None:
            return None
        return self.input_tokens * cin / 1e6 + self.output_tokens * cout / 1e6

    def with_result(self, stage_key: str, result: StageResult, conversation_id: str) -> PipelineState:
        results = dict(self.results)
        results[stage_key] = result
        conversations = dict(self.conversations)
        conversations[stage_key] = conversation_id
        return replace(self, results=results, conversations=conversations)

    def with_verdict(self, verdict: str) -> PipelineState:
        return replace(self, verdict=verdict)

    def finish(self) -> PipelineState:
        return replace(self, finished_at=time.time())
