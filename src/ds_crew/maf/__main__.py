"""CLI entrypoint: `ds-crew-maf`."""

from __future__ import annotations

import argparse
import asyncio
import sys

from agent_framework import FileCheckpointStorage
from azure.identity import DefaultAzureCredential

from ds_crew import settings
from ds_crew.foundry.stages import STAGES
from ds_crew.maf.host import (
    PreflightError,
    auto_decider,
    auto_verdict_collector,
    create_run,
    describe_checkpoints,
    drive,
    interactive_decider,
    interactive_verdict_collector,
    preflight,
    summarize,
)
from ds_crew.maf.state import PipelineState
from ds_crew.maf.telemetry import setup_observability
from ds_crew.maf.transport_foundry import FoundryTransport
from ds_crew.maf.workflow import WORKFLOW_NAME, build_workflow

# module:qualname strings FileCheckpointStorage's restricted unpickler must
# trust beyond its own built-in/agent_framework/openai allow-list -- these are
# the only application types a checkpoint ever actually embeds (see
# ds_crew.maf.state's module docstring on why PipelineState is fully
# serializable in the first place).
_CHECKPOINT_TYPES = ["ds_crew.maf.state:PipelineState", "ds_crew.foundry.runner:StageResult"]


def main(argv: list[str] | None = None) -> int:
    # Before any agent/workflow spans can be created, so nothing from this
    # run is dropped -- a no-op unless APPLICATIONINSIGHTS_CONNECTION_STRING
    # is set (see ds_crew.maf.telemetry).
    setup_observability()

    parser = argparse.ArgumentParser(
        prog="ds-crew-maf",
        description=(
            "Run the DS-Crew pipeline across the eight agents hosted in Azure AI "
            "Foundry, via Microsoft Agent Framework. Ordering is deterministic and "
            "lives in ds_crew.foundry.stages; the four gated tools pause for a human "
            "unless --auto-approve is given. Every run is checkpointed at each stage "
            "boundary -- see --resume and --list-checkpoints."
        ),
    )
    # Not required=True: --viz/--list-checkpoints only need the workflow's
    # static topology or its checkpoint store, not a run to drive, and must
    # work standalone.
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", help="Dataset to create a new run from.")
    source.add_argument("--run-id", help="Drive an existing run instead of creating one.")
    source.add_argument(
        "--resume",
        metavar="CHECKPOINT_ID",
        help=(
            "Resume a previously checkpointed run instead of starting one -- see "
            "--list-checkpoints. The whole PipelineState is restored from the "
            "checkpoint, so --csv/--target/--task/--metric/--run-id do not apply; "
            "--auto-approve/--verdict still govern whatever stages remain."
        ),
    )
    parser.add_argument("--target", help="Target column. Required with --csv.")
    parser.add_argument("--task", default="auto", choices=("classification", "regression", "auto"))
    parser.add_argument("--metric", default=None, help="Optimization metric. Defaults per task.")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help=(
            "Approve all four gated tools without asking, for an unattended "
            "end-to-end run. Defaults to the AUTO_APPROVE setting."
        ),
    )
    parser.add_argument(
        "--verdict",
        default="",
        help=(
            "The sign-off text handed to the evaluator at the final stage. With "
            "--auto-approve and no verdict, the agent is told no human reviewed the "
            "run, and will record a rejection."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the tool-service reachability check. Rarely a good idea.",
    )
    parser.add_argument(
        "--viz",
        default=None,
        help="Write a Mermaid diagram of the workflow topology to this path and exit.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=".ds_crew_checkpoints",
        help=(
            "Where stage-boundary checkpoints are written (default: %(default)s). "
            "Every run is checkpointed; a mid-stage crash replays that stage on "
            "resume, which is safe here since the gated tools are one-shot and "
            "refuse a genuine repeat cleanly."
        ),
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List saved checkpoints (run, stages completed, timestamp) and exit.",
    )
    args = parser.parse_args(argv)

    if args.viz:
        from ds_crew.maf.viz import write_mermaid

        write_mermaid(args.viz)
        print(f"Wrote workflow diagram to {args.viz}")
        return 0

    if not args.list_checkpoints and not args.csv and not args.run_id and not args.resume:
        parser.error("one of the arguments --csv --run-id --resume is required")

    if args.csv and not args.target:
        parser.error("--target is required with --csv")

    # Constructed only once a mode that actually needs it survives validation
    # above -- FileCheckpointStorage creates --checkpoint-dir as a side effect
    # of construction, which a bare/invalid invocation shouldn't trigger.
    checkpoint_storage = FileCheckpointStorage(
        args.checkpoint_dir, allowed_checkpoint_types=_CHECKPOINT_TYPES
    )

    if args.list_checkpoints:
        rows = asyncio.run(describe_checkpoints(checkpoint_storage, workflow_name=WORKFLOW_NAME))
        if not rows:
            print(f"No checkpoints found in {args.checkpoint_dir}")
            return 0
        for row in rows:
            progress = (
                f"{row['stages_done']}/{len(STAGES)} stages"
                if row["stages_done"] is not None
                else "terminal (no stage left pending)"
            )
            print(f"{row['checkpoint_id']}  run={row['run_id'] or '?'}  {progress}  {row['timestamp']}")
        return 0

    if not args.skip_preflight:
        try:
            preflight()
        except PreflightError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    auto_mode = args.auto_approve or settings.AUTO_APPROVE
    verdict = args.verdict
    if auto_mode and not verdict:
        # Unattended with nothing to say: recording a sign-off nobody gave
        # would be worse than recording an explicit, honest rejection.
        verdict = (
            "No human reviewed this run; it was executed unattended. Record it as NOT "
            "approved."
        )
        print(
            "Note: --auto-approve without --verdict records the run as rejected, because "
            "no human signed it off. Pass --verdict to state a decision.\n"
        )

    decide = auto_decider() if auto_mode else interactive_decider()
    collect_verdict = auto_verdict_collector() if auto_mode else interactive_verdict_collector()

    transport = FoundryTransport(
        project_endpoint=settings.AZURE_FOUNDRY_PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )
    workflow = build_workflow(
        transport=transport,
        decide=decide,
        collect_verdict=collect_verdict,
        checkpoint_storage=checkpoint_storage,
    )

    try:
        if args.resume:
            print(f"Resuming from checkpoint {args.resume}\n")
            final_state = asyncio.run(drive(workflow, checkpoint_id=args.resume))
        else:
            run_id = args.run_id or create_run(args.csv, args.target, task=args.task, metric=args.metric)
            if not args.run_id:
                print(f"Created run {run_id}")
            state = PipelineState(run_id=run_id, verdict=verdict)
            print(
                f"\nRun {run_id} -- {len(STAGES)} stages, "
                f"{'auto-approving' if auto_mode else 'human'} gates\n"
            )
            final_state = asyncio.run(drive(workflow, state))
    except KeyboardInterrupt:
        print(
            "\nInterrupted. The run keeps whatever stages already applied, and the last "
            f"checkpoint is in {args.checkpoint_dir} -- see --list-checkpoints.",
            file=sys.stderr,
        )
        return 130

    summarize(final_state)
    return 0 if len(final_state.results) == len(STAGES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
