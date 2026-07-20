"""Shared in-process run state.

Tools never put full DataFrames into LLM context -- they read/mutate the
actual objects held here and return compact JSON-safe summaries. `run_id` is
bound into each tool's constructor at crew-build time (see crew.py), never
exposed as an LLM-callable argument, so a hallucinating or prompt-injected
agent can never address another run's data.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ds_crew.schemas import EvaluationReport, HpoResult, Leaderboard, TaskType


@dataclass(eq=False)
class RunState:
    run_id: str
    df: pd.DataFrame
    target: str
    task_type: TaskType | None = None
    test_size: float = 0.2
    # The actual MLflow run id (set once in main.py after mlflow.start_run()).
    # Logging helpers require this explicitly rather than relying on
    # mlflow.active_run(), which is thread-local and invisible from the
    # worker thread CrewAI executes tool calls in -- see logging_tools.py.
    mlflow_run_id: str | None = None

    # Set once the corresponding execute-stage tool has successfully mutated the
    # run. Some models fragment a single large structured tool call into several
    # partial invocations (observed live: an LLM split one CleaningPlan call
    # into 10+ calls, each carrying a different subset of fields); since every
    # field on CleaningPlan/FeatureEngineeringPlan has a default, each fragment
    # would otherwise "succeed" silently instead of being rejected. These flags
    # let the execute-stage tools refuse a second application outright.
    cleaning_applied: bool = False
    features_applied: bool = False

    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None
    feature_names: list[str] | None = None
    fitted_transformer: Any | None = None

    leaderboard: Leaderboard | None = None
    hpo_results: dict[str, HpoResult] = field(default_factory=dict)
    fitted_models: dict[str, Any] = field(default_factory=dict)
    evaluation_reports: dict[str, EvaluationReport] = field(default_factory=dict)

    history: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    artifacts_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.artifacts_dir = Path("runs") / self.run_id
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def record(self, stage: str, action: str, detail: dict[str, Any]) -> None:
        self.history.append({"stage": stage, "action": action, "detail": detail})


class DataStore:
    """Process-wide singleton registry of RunState, keyed by run_id."""

    _instance: "DataStore | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "DataStore":
        with cls._instance_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._runs = {}
                cls._instance = inst
            return cls._instance

    def create_run(self, run_id: str, df: pd.DataFrame, target: str, **kwargs: Any) -> RunState:
        state = RunState(run_id=run_id, df=df, target=target, **kwargs)
        self._runs[run_id] = state
        return state

    def get(self, run_id: str) -> RunState:
        if run_id not in self._runs:
            raise KeyError(f"Unknown run_id '{run_id}'. Known runs: {list(self._runs)}")
        return self._runs[run_id]

    def drop(self, run_id: str) -> None:
        self._runs.pop(run_id, None)


def get_data_store() -> DataStore:
    return DataStore()


def reset_data_store() -> None:
    """Test-only helper: clears the singleton so tests don't leak state across runs."""
    DataStore._instance = None
