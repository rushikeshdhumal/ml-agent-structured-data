"""CLI entrypoint. One invocation = one dataset = one CrewAI run = one MLflow run.

Raw data acquisition is out of scope for this system -- the caller always
supplies the dataset path and target column explicitly via CLI flags.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import uuid
from pathlib import Path

import mlflow
import pandas as pd

from ds_crew import settings
from ds_crew.crew import DsCrew, active_llm_provider
from ds_crew.state import get_data_store
from ds_crew.tools.eda_tools import infer_task_type
from ds_crew.tools.logging_tools import log_llm_usage
from ds_crew.tools.model_tools import ALLOWED_METRICS, METRIC_BY_TASK

# Windows' default stdout/stderr encoding for a non-console-attached process
# (e.g. output redirected to a file/pipe) is often cp1252 ("charmap"), which
# can't encode the emoji CrewAI's verbose console output uses -- that crashes
# every internal event-bus print handler. Force UTF-8 unconditionally so the
# crew's live-progress output never silently breaks.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass


def _dataset_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ds-crew",
        description="Run the structured-data ML crew on an explicitly provided dataset.",
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to a CSV dataset.")
    parser.add_argument("--target", required=True, help="Name of the target column.")
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "auto"],
        default="auto",
        help="Task type. 'auto' infers it from the target column.",
    )
    parser.add_argument(
        "--run-name", default=None, help="MLflow run name (defaults to the generated run id)."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=settings.DEFAULT_TEST_SIZE,
        help="Held-out test fraction, used by cleaning's train/test split.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help=(
            "Optimization metric driving cross-validation, HPO, ensembling, and evaluation. "
            "Defaults to a task-appropriate metric; validated against the allowed set for "
            "the resolved task type. In an interactive run, the metric-selection gate lets a "
            "human override this before model selection regardless of what's passed here."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.data.exists():
        parser.error(f"--data path does not exist: {args.data}")

    if not 0.0 < args.test_size < 1.0:
        parser.error(f"--test-size must be strictly between 0 and 1, got {args.test_size}")

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        parser.error(f"--target '{args.target}' not found in columns: {list(df.columns)}")

    task_type = args.task if args.task != "auto" else infer_task_type(df[args.target])

    if args.metric and args.metric not in ALLOWED_METRICS[task_type]:
        parser.error(
            f"--metric '{args.metric}' not allowed for task type '{task_type}': "
            f"{sorted(ALLOWED_METRICS[task_type])}"
        )

    run_id = uuid.uuid4().hex[:12]

    state = get_data_store().create_run(run_id, df, target=args.target)
    state.task_type = task_type
    state.test_size = args.test_size
    # The metric-selection gate (propose_metric_task/set_metric_task) can
    # still override this before model selection runs in an interactive run;
    # this just seeds a sensible starting point for headless/AUTO_APPROVE runs.
    state.metric_name = args.metric or METRIC_BY_TASK[task_type]

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=args.run_name or run_id) as active_run:
        # Tool calls happen from a worker thread CrewAI spawns internally, where
        # mlflow.active_run() is invisible (it's thread-local) -- logging_tools.py
        # needs the real run id explicitly rather than relying on ambient state.
        state.mlflow_run_id = active_run.info.run_id
        mlflow.set_tags(
            {
                "run_id": run_id,
                "dataset_hash": _dataset_hash(args.data),
                "task_type": task_type,
                "auto_approve": str(settings.AUTO_APPROVE).lower(),
                # Which routing branch served this run, alongside the model id
                # itself. Token counts and estimated_cost_usd are already logged
                # (see log_llm_usage); without knowing which provider produced
                # them, runs against NVIDIA NIM and Azure Foundry are not
                # comparable, which is the main reason to be able to switch.
                "llm_provider": active_llm_provider(),
                "model": settings.MODEL,
            }
        )
        mlflow.log_params(
            {
                "data_path": str(args.data),
                "target": args.target,
                "test_size": args.test_size,
                "n_rows": len(df),
                "n_cols": len(df.columns),
            }
        )
        # Built before the try so the `finally` can still read usage off it if
        # kickoff raises -- a run that burned tokens and then crashed is exactly
        # the one whose spend needs recording.
        crew = DsCrew(run_id=run_id).crew()
        started = time.perf_counter()
        try:
            crew.kickoff(inputs={"run_id": run_id, "target": args.target})
            mlflow.set_tag("status", "completed")
        except Exception:
            mlflow.set_tag("status", "failed")
            raise
        finally:
            log_llm_usage(state.mlflow_run_id, crew, time.perf_counter() - started)

    print(f"Run {run_id} complete. MLflow tracking URI: {settings.MLFLOW_TRACKING_URI}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
