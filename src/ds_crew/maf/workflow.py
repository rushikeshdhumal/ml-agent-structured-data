"""Assemble the MAF workflow graph from `stages.STAGES` -- never by hand.

Nine hand-written `add_edge` calls would be a second, driftable source of
truth for an ordering `stages.py` exists specifically to make deterministic
and reviewable. The one structural addition beyond a straight chain is the
human-verdict node, spliced in between `explanation` and `finalize` to mirror
`run_pipeline`'s live verdict collection immediately before the finalize stage.

`WORKFLOW_NAME` must stay a fixed constant, not derived per call: a
`WorkflowBuilder` left to name itself gets a fresh random
`WorkflowBuilder-<uuid4>` on every construction (verified against the
installed `agent_framework`, 2026-08-26), and `FileCheckpointStorage` scopes
every checkpoint by that name. A random name would make a checkpoint saved by
one process invocation unreachable from the next, which defeats `--resume`
entirely -- the whole point is resuming in a *new* process after a crash.
"""

from __future__ import annotations

from typing import Any

from agent_framework import Workflow, WorkflowBuilder

from ds_crew.foundry.stages import STAGES, Stage
from ds_crew.maf.executors import HumanVerdictExecutor, StageExecutor, VerdictCollector
from ds_crew.maf.transport import Decider, StageTransport

# Every run of this pipeline shares one checkpoint namespace, regardless of
# run_id -- see the module docstring for why this can't be left to default.
WORKFLOW_NAME = "ds-crew-maf"


def build_workflow(
    *,
    transport: StageTransport,
    decide: Decider,
    collect_verdict: VerdictCollector,
    stages: tuple[Stage, ...] = STAGES,
    checkpoint_storage: Any | None = None,
    log: Any = print,
) -> Workflow:
    total = len(stages)
    nodes = [
        StageExecutor(
            stage,
            transport,
            decide,
            index=i,
            total=total,
            is_last=(stage.key == stages[-1].key),
            log=log,
        )
        for i, stage in enumerate(stages, 1)
    ]
    verdict_node = HumanVerdictExecutor(collect_verdict)

    builder = WorkflowBuilder(
        name=WORKFLOW_NAME, start_executor=nodes[0], checkpoint_storage=checkpoint_storage
    )
    for current, following in zip(nodes, nodes[1:]):
        if following.id == "finalize":
            builder.add_edge(current, verdict_node)
            builder.add_edge(verdict_node, following)
        else:
            builder.add_edge(current, following)

    return builder.build()
