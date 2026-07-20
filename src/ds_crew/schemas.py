"""Shared Pydantic models for propose/execute plans and reports.

Every plan/report that a guardrail may need to inspect carries `run_id` as
its first field, since a CrewAI Task guardrail receives nothing but the
task's `TaskOutput` -- `run_id` is its only way back into the DataStore.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field

TaskType = Literal["classification", "regression"]


def _none_if_null_string(v: Any) -> Any:
    """Some models emit the literal string "null" (observed live: NVIDIA's
    z-ai/glm-5.2 via structured tool-call arguments) instead of a true JSON
    null for an Optional field. Coerce it back to None before type
    validation, rather than letting Pydantic reject it as a bad string/int/etc.
    """
    if isinstance(v, str) and v.strip().lower() in ("null", "none"):
        return None
    return v


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_pct: float
    n_unique: int
    is_constant: bool
    is_id_like: bool
    sample_values: list[str] = Field(default_factory=list)


class EdaReport(BaseModel):
    run_id: str
    n_rows: int
    n_cols: int
    target: str
    task_type: TaskType
    columns: list[ColumnProfile]
    correlations_with_target: dict[str, float] = Field(default_factory=dict)
    near_duplicate_column_pairs: list[tuple[str, str]] = Field(default_factory=list)
    class_balance: Annotated[
        dict[str, float] | None, BeforeValidator(_none_if_null_string)
    ] = None
    leakage_flags: list[str] = Field(default_factory=list)
    truncated: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

MissingStrategy = Literal[
    "drop_rows", "mean", "median", "mode", "constant", "ffill", "bfill", "knn"
]
OutlierStrategy = Literal["none", "iqr_clip", "zscore_clip", "drop"]


class ColumnCleaningAction(BaseModel):
    column: str
    missing_strategy: Annotated[
        MissingStrategy | None, BeforeValidator(_none_if_null_string)
    ] = None
    constant_fill_value: Annotated[
        str | float | None, BeforeValidator(_none_if_null_string)
    ] = None
    outlier_strategy: OutlierStrategy = "none"
    dtype_cast: Annotated[
        Literal["int", "float", "str", "category", "bool", "datetime"] | None,
        BeforeValidator(_none_if_null_string),
    ] = None


class CleaningPlan(BaseModel):
    run_id: str
    actions: list[ColumnCleaningAction] = Field(default_factory=list)
    drop_duplicate_rows: bool = False
    columns_to_drop: list[str] = Field(default_factory=list)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

EncodingStrategy = Literal["onehot", "ordinal", "target_mean", "frequency", "none"]
ScalingStrategy = Literal["standard", "minmax", "robust", "none"]


class ColumnFeaturePlan(BaseModel):
    column: str
    encoding: EncodingStrategy = "none"
    scaling: ScalingStrategy = "none"
    datetime_decompose: bool = False


class FeatureEngineeringPlan(BaseModel):
    run_id: str
    column_plans: list[ColumnFeaturePlan] = Field(default_factory=list)
    features_to_drop: list[str] = Field(default_factory=list)
    feature_selection_method: Literal["none", "mutual_info", "variance_threshold"] = "none"
    top_k: Annotated[int | None, BeforeValidator(_none_if_null_string)] = None
    rationale: str = ""


# ---------------------------------------------------------------------------
# Model selection / HPO
# ---------------------------------------------------------------------------


class ModelCandidateResult(BaseModel):
    model_name: str
    cv_mean_score: float
    cv_std: float
    fit_time_s: float


class Leaderboard(BaseModel):
    run_id: str
    task_type: TaskType
    metric_name: str
    candidates: list[ModelCandidateResult]
    warnings: list[str] = Field(default_factory=list)


class HpoResult(BaseModel):
    model_name: str
    best_params: dict
    best_score: float
    n_trials: int


class HpoResults(BaseModel):
    run_id: str
    results: list[HpoResult]


# ---------------------------------------------------------------------------
# Evaluation / sign-off
# ---------------------------------------------------------------------------


class EvaluationReport(BaseModel):
    model_name: str
    metrics: dict[str, float]
    confusion_matrix: Annotated[
        list[list[int]] | None, BeforeValidator(_none_if_null_string)
    ] = None
    feature_importances: Annotated[
        dict[str, float] | None, BeforeValidator(_none_if_null_string)
    ] = None
    leakage_suspicion: bool = False
    notes: str = ""


class EvaluationBundle(BaseModel):
    run_id: str
    reports: list[EvaluationReport]


class FinalSignOff(BaseModel):
    run_id: str
    selected_model: str
    approved: bool
    human_feedback: str = ""
