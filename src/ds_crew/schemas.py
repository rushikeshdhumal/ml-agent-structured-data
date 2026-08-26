"""Shared Pydantic models for propose/execute plans and reports.

Every plan/report that a guardrail may need to inspect carries `run_id` as
its first field, since a CrewAI Task guardrail receives nothing but the
task's `TaskOutput` -- `run_id` is its only way back into the DataStore.
"""

from __future__ import annotations

import re
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


def _canonical(v: Any) -> str:
    return re.sub(r"[\s_\-]+", "", str(v)).lower()


def enum_alias(*allowed: str) -> BeforeValidator:
    """Accept an enum value written with any separator style.

    Observed live: a Foundry agent given the inlined schema for
    `apply_feature_plan` got every field name right and then sent
    `encoding: "one_hot"` against a `Literal["onehot", ...]`. The model was not
    really wrong. `one_hot` is the near-universal spelling -- sklearn's
    `OneHotEncoder`, "one-hot encoding" in every textbook -- and `onehot` is the
    idiosyncratic one. A strong prior beats an unusual literal, and it will beat
    it again on the next model and the next run.

    Matching on the value with separators and case stripped accepts `one_hot`,
    `one-hot` and `OneHot` for `onehot`, `min_max` for `minmax`, `drop-rows` for
    `drop_rows`, and so on, without inventing new vocabulary. This cannot merge
    two distinct options: it only unifies strings that are already identical
    apart from separators, and `test_enum_aliases_cannot_collide` asserts no
    pair in this module collides that way.

    Deliberately not a synonym table. `forward_fill` for `ffill` would be a
    guess about intent; this is only a spelling normalization.
    """
    lookup = {_canonical(a): a for a in allowed}

    def normalize(v: Any) -> Any:
        if isinstance(v, str):
            return lookup.get(_canonical(v), v)
        return v

    return BeforeValidator(normalize)


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

MISSING_STRATEGIES = ("drop_rows", "mean", "median", "mode", "constant", "ffill", "bfill", "knn")
OUTLIER_STRATEGIES = ("none", "iqr_clip", "zscore_clip", "drop")
DTYPE_CASTS = ("int", "float", "str", "category", "bool", "datetime")

MissingStrategy = Literal[MISSING_STRATEGIES]  # type: ignore[valid-type]
OutlierStrategy = Literal[OUTLIER_STRATEGIES]  # type: ignore[valid-type]


class ColumnCleaningAction(BaseModel):
    column: str
    missing_strategy: Annotated[
        MissingStrategy | None,
        BeforeValidator(_none_if_null_string),
        enum_alias(*MISSING_STRATEGIES),
    ] = None
    constant_fill_value: Annotated[
        str | float | None, BeforeValidator(_none_if_null_string)
    ] = None
    outlier_strategy: Annotated[
        OutlierStrategy, enum_alias(*OUTLIER_STRATEGIES)
    ] = "none"
    dtype_cast: Annotated[
        Literal[DTYPE_CASTS] | None,  # type: ignore[valid-type]
        BeforeValidator(_none_if_null_string),
        enum_alias(*DTYPE_CASTS),
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

ENCODING_STRATEGIES = ("onehot", "ordinal", "target_mean", "frequency", "none")
SCALING_STRATEGIES = ("standard", "minmax", "robust", "none")
SELECTION_METHODS = ("none", "mutual_info", "variance_threshold")

EncodingStrategy = Literal[ENCODING_STRATEGIES]  # type: ignore[valid-type]
ScalingStrategy = Literal[SCALING_STRATEGIES]  # type: ignore[valid-type]


class ColumnFeaturePlan(BaseModel):
    column: str
    encoding: Annotated[EncodingStrategy, enum_alias(*ENCODING_STRATEGIES)] = "none"
    scaling: Annotated[ScalingStrategy, enum_alias(*SCALING_STRATEGIES)] = "none"
    datetime_decompose: bool = False


class FeatureEngineeringPlan(BaseModel):
    run_id: str
    column_plans: list[ColumnFeaturePlan] = Field(default_factory=list)
    features_to_drop: list[str] = Field(default_factory=list)
    feature_selection_method: Annotated[
        Literal[SELECTION_METHODS],  # type: ignore[valid-type]
        enum_alias(*SELECTION_METHODS),
    ] = "none"
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
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Metric selection
# ---------------------------------------------------------------------------


class MetricChoice(BaseModel):
    run_id: str
    metric: str
    rationale: str = ""


# ---------------------------------------------------------------------------
# Ensembling
# ---------------------------------------------------------------------------

EnsembleStrategy = Literal["equal_voting", "weighted_voting", "stacking", "greedy"]


class EnsembleReport(BaseModel):
    run_id: str
    ensemble_name: str = "ensemble"
    strategy: EnsembleStrategy
    member_models: list[str]
    weights: Annotated[list[float] | None, BeforeValidator(_none_if_null_string)] = None
    final_estimator: Annotated[str | None, BeforeValidator(_none_if_null_string)] = None
    metric_name: str
    cv_mean_score: float
    best_single_model: str
    best_single_cv_score: float
    improved_over_best_single: bool
    warnings: list[str] = Field(default_factory=list)
    notes: str = ""


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


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

# How a report's attributions were produced. Permutation importance is always
# computed (it is the only method that works uniformly across every registry
# model AND the VotingClassifier/StackingClassifier ensemble); SHAP is layered
# on top only where it is exact and cheap. "permutation_only" therefore covers
# both the models with no cheap exact explainer (knn, ensemble) and any model
# whose SHAP call failed and fell back -- `warnings` distinguishes the two.
AttributionMethod = Literal["shap_tree", "shap_linear", "permutation_only"]


class FeatureAttribution(BaseModel):
    """One engineered feature's contribution. `source_column` is the original
    dataset column it came from -- feature_tools.py names each ColumnTransformer
    entry after its source column, so an engineered name is always
    "{source_column}__{...}" and the rollup is exact, not a guess.
    """

    feature: str
    source_column: str
    permutation_importance: float
    permutation_std: float
    mean_abs_shap: Annotated[float | None, BeforeValidator(_none_if_null_string)] = None
    direction: Annotated[
        Literal["increases", "decreases", "mixed"] | None,
        BeforeValidator(_none_if_null_string),
    ] = None


class LocalExplanation(BaseModel):
    """A single row's signed attributions. `row_position` is a positional index
    into the explained X_test sample -- X_test is a numpy array by this stage,
    so there is no DataFrame label to refer to.
    """

    kind: Literal["confident_correct", "confident_wrong", "most_uncertain"]
    row_position: int
    actual: str
    predicted: str
    predicted_proba: Annotated[float | None, BeforeValidator(_none_if_null_string)] = None
    top_contributions: dict[str, float] = Field(default_factory=dict)


class ExplanationReport(BaseModel):
    run_id: str
    model_name: str
    metric_name: str
    explained_on: Literal["test", "train"] = "test"
    n_rows_explained: int
    attribution_method: AttributionMethod
    top_features: list[FeatureAttribution] = Field(default_factory=list)
    # Permutation importance rolled up from engineered features to the original
    # dataset columns and normalized to sum to 1.0 -- the form a human actually
    # reasons about ("cat_a carries 22% of the signal"), rather than 40 one-hot
    # columns each carrying a sliver.
    column_importance: dict[str, float] = Field(default_factory=dict)
    local_explanations: list[LocalExplanation] = Field(default_factory=list)
    surrogate_rules: Annotated[str | None, BeforeValidator(_none_if_null_string)] = None
    surrogate_fidelity: Annotated[float | None, BeforeValidator(_none_if_null_string)] = None
    unused_features: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: str = ""


class ExplanationBundle(BaseModel):
    run_id: str
    reports: list[ExplanationReport]


# ---------------------------------------------------------------------------
# Sign-off
# ---------------------------------------------------------------------------


class FinalSignOff(BaseModel):
    run_id: str
    selected_model: str
    approved: bool
    human_feedback: str = ""
