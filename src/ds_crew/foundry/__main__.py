"""CLI entrypoint: `ds-crew-foundry`."""

from __future__ import annotations

import argparse
import sys

from ds_crew import settings
from ds_crew.foundry.orchestrator import (
    PreflightError,
    build_project_client,
    create_run,
    preflight,
    run_pipeline,
)
from ds_crew.foundry.stages import STAGES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ds-crew-foundry",
        description=(
            "Run the DS-Crew pipeline across the eight agents hosted in Azure AI Foundry. "
            "Ordering is deterministic and lives in ds_crew.foundry.stages; the four gated "
            "tools pause for a human unless --auto-approve is given."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Dataset to create a new run from.")
    source.add_argument("--run-id", help="Drive an existing run instead of creating one.")

    parser.add_argument("--target", help="Target column. Required with --csv.")
    parser.add_argument("--task", default="auto", choices=("classification", "regression", "auto"))
    parser.add_argument("--metric", default=None, help="Optimization metric. Defaults per task.")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help=(
            "Approve all four gated tools without asking, for an unattended end-to-end run. "
            "Defaults to the AUTO_APPROVE setting."
        ),
    )
    parser.add_argument(
        "--verdict",
        default="",
        help=(
            "The sign-off text handed to the evaluator at the final stage. With "
            "--auto-approve and no verdict, the agent is told no human reviewed the run, "
            "and will record a rejection."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the tool-service reachability check. Rarely a good idea.",
    )
    args = parser.parse_args(argv)

    if args.csv and not args.target:
        parser.error("--target is required with --csv")

    try:
        if not args.skip_preflight:
            preflight()
    except PreflightError as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2

    auto = args.auto_approve or None
    verdict = args.verdict
    # Interactively (auto False) with no --verdict, this stays empty on purpose:
    # run_pipeline collects the real sign-off live, after the explanation
    # prints, rather than requiring one written blind before the run starts.
    if (args.auto_approve or settings.AUTO_APPROVE) and not verdict:
        # finalize_run defaults to approved=false without an unambiguous human
        # statement, and that is correct: a sign-off nobody gave should not be
        # recorded. Say so plainly rather than letting the run look broken.
        verdict = (
            "No human reviewed this run; it was executed unattended. Record it as NOT "
            "approved."
        )
        print(
            "Note: --auto-approve without --verdict records the run as rejected, because "
            "no human signed it off. Pass --verdict to state a decision.\n"
        )

    run_id = args.run_id or create_run(
        args.csv, args.target, task=args.task, metric=args.metric
    )
    if not args.run_id:
        print(f"Created run {run_id}")

    try:
        client = build_project_client()
    except ImportError:
        print(
            "The Foundry extra is not installed. Run:\n"
            "  uv sync --extra dev --extra service --extra foundry",
            file=sys.stderr,
        )
        return 2

    try:
        report = run_pipeline(run_id, project_client=client, auto_approve=auto, verdict=verdict)
    except KeyboardInterrupt:
        print("\nInterrupted. The run keeps whatever stages already applied.", file=sys.stderr)
        return 130

    completed = len(report.results)
    return 0 if completed == len(STAGES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
