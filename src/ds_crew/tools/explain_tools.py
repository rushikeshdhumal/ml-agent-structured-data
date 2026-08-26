"""Deterministic model explainability.

Answers "what did this model actually learn?" for a human deciding whether to
sign the model off. Like every other stage here, the agent never produces an
attribution itself -- it calls this tool once and narrates what comes back.

Nothing currently verifies that narration against what this module actually
computed. CrewAI's `guardrails.make_explanation_grounded_guardrail` used to
check the reported model names against `state.explanation_reports`, but that
callback only exists inside CrewAI's Task-retry loop, which this branch no
longer has, and Foundry agents have no equivalent hook to attach one to. This
is the gap Phase 9 of the Foundry plan means to close with a custom
evaluator; until then, a fabricated-but-plausible attribution reaching the
sign-off gate is a real, open risk, not a hypothetical one.

Two layers, deliberately:

* **Permutation importance is the floor, always computed.** It is the only
  method that works uniformly across this project's whole registry *and* the
  VotingClassifier/StackingClassifier the ensembler builds. That matters more
  than it sounds: `EvaluationReport.feature_importances` reads
  `feature_importances_`/`coef_`, which do not exist on a VotingClassifier or
  a KNeighborsClassifier at all -- so the ensemble, frequently the model
  recommended for sign-off, previously reached the human with no attribution
  data whatsoever.
* **SHAP on top, only where it is exact and cheap** -- TreeExplainer for the
  three boosting models, LinearExplainer for the linear ones. Every SHAP call
  is isolated in try/except and degrades to permutation-only with a recorded
  warning, mirroring `model_tools.run_cross_validation`'s per-candidate
  isolation. That is not boilerplate caution: `_XGBClassifierWithLabelEncoding`
  has already broken two libraries that pattern-match on estimator type
  (`mlflow.xgboost.save_model` raises on it outright), so a third-party
  explainer meeting it is a live risk, not a hypothetical one.

X_test relationship to the "scored exactly once" invariant: this module reads
X_test, but only read-only, only after `EvaluateModelsTool` has finished (it
refuses to run otherwise), and its output feeds only the terminal human
sign-off gate -- never model selection, tuning, or ensembling. Nothing here
can be optimized against. See eval_tools.py's docstring.

Class order is derived from `sorted(np.unique(y_train))`, never
`model.classes_`: `_XGBClassifierWithLabelEncoding.classes_` reports the
internal integer encoding XGBoost requires at fit time, not the original label
space, and it provably cannot be patched without breaking `fit()`.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from ds_crew.tools.base import Tool
from pydantic import BaseModel, Field
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from ds_crew import settings
from ds_crew.schemas import (
    AttributionMethod,
    ExplanationBundle,
    ExplanationReport,
    FeatureAttribution,
    LocalExplanation,
    TaskType,
)
from ds_crew.state import get_data_store
from ds_crew.tools.logging_tools import log_json_artifact, log_stage_metrics
from ds_crew.tools.model_tools import resolve_cv_scorer

# Which explainer each registry model gets. Absent from this map (knn, and the
# "ensemble" pseudo-candidate) means permutation-only: SHAP's model-agnostic
# KernelExplainer would work, but it is O(n_rows * n_features * n_background)
# model calls, which blows the bounded-compute budget every other stage here
# respects. Permutation importance already covers those models correctly.
_SHAP_METHOD_BY_MODEL: dict[str, AttributionMethod] = {
    "xgboost": "shap_tree",
    "lightgbm": "shap_tree",
    "catboost": "shap_tree",
    "logistic_regression": "shap_linear",
    "ridge": "shap_linear",
    "elastic_net": "shap_linear",
}


def shap_method_for(model_name: str, n_classes: int | None) -> AttributionMethod | None:
    """Which SHAP explainer is safe for this model, or None for permutation-only.

    The catboost/multiclass exclusion is a hard pre-emptive guard, deliberately
    *not* a try/except: shap 0.52.0's TreeExplainer **segfaults** on a
    multiclass CatBoostClassifier (verified on this stack -- exit code 139,
    reproducible at n_classes=3, while n_classes=2 is fine). A segfault
    terminates the interpreter, so no exception handler can degrade it
    gracefully; it would take down the entire crew run mid-pipeline and lose
    every stage's results. The only safe handling is never to make the call.

    xgboost/lightgbm multiclass were both verified working, so this is
    specifically a catboost limitation, not a general multiclass one.
    """
    method = _SHAP_METHOD_BY_MODEL.get(model_name)
    if method is None:
        return None
    if model_name == "catboost" and n_classes is not None and n_classes > 2:
        return None
    return method

# Below this absolute mean-|SHAP| share, a feature's sign is not worth
# reporting a direction for -- it would read as a claim about behavior the
# model does not really have.
_DIRECTION_MIXED_RATIO = 0.6


def source_column_of(feature_name: str) -> str:
    """Map an engineered feature back to the dataset column it came from.

    `feature_tools.build_transformer` creates one ColumnTransformer entry per
    column and names that entry after the column itself, so
    `get_feature_names_out()` always yields "{source_column}__{engineered}"
    (e.g. "cat_a__cat_a_red", "num_a__num_a"). The rollup is therefore exact
    rather than a heuristic. `select_features` passes names through verbatim,
    so the no-separator branch is defensive only.
    """
    return feature_name.split("__", 1)[0] if "__" in feature_name else feature_name


def _subsample(X: np.ndarray, y: pd.Series, max_rows: int, seed: int):
    """Seeded row subsample so explanation cost stays bounded on large test
    splits. Returns the arrays unchanged when they already fit.
    """
    n = X.shape[0]
    if n <= max_rows:
        return X, y, np.arange(n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_rows, replace=False))
    return X[idx], y.iloc[idx], idx


def compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: pd.Series,
    metric: str,
    y_train: pd.Series,
    n_repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Model-agnostic importance on held-out rows.

    `resolve_cv_scorer` is reused (not reimplemented) so this stage scores on
    exactly the metric the rest of the run optimized -- including the
    roc_auc/roc_auc_ovr distinction. It is resolved against `y_train`, the
    canonical class-order source everywhere in this codebase, not against the
    test labels, whose class count could differ on a small split.
    """
    result = permutation_importance(
        model,
        X,
        y,
        scoring=resolve_cv_scorer(metric, y_train),
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    return result.importances_mean, result.importances_std


def _normalize_shap_values(values: np.ndarray, n_features: int) -> np.ndarray:
    """Reduce whatever shape SHAP returned to a signed (n_rows, n_features)
    array.

    Shapes vary by shap version and task: binary classification comes back as
    either (n, f) or (n, f, 2), multiclass as (n, f, n_classes). For the 3-D
    cases we take the last class's column -- for binary that is the positive
    class under the sorted-class convention this codebase uses everywhere, so
    a positive value means "pushes toward the positive class". Multiclass has
    no single meaningful signed direction, so callers use mean(|.|) across
    classes for the global number and only surface signed values when binary.
    """
    values = np.asarray(values)
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        if values.shape[1] == n_features:
            return values[:, :, -1]
        # Some versions emit (n_classes, n, f).
        return values[-1]
    raise ValueError(f"Unexpected SHAP value shape {values.shape}")


def _mean_abs_shap(values: np.ndarray, n_features: int) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 3:
        axes = (0, 2) if values.shape[1] == n_features else (0, 1)
        return np.abs(values).mean(axis=axes)
    return np.abs(values).mean(axis=0)


def compute_shap_values(
    model: Any, method: AttributionMethod, X: np.ndarray, seed: int
) -> np.ndarray:
    """Raw SHAP values for a model whose explainer `shap_method_for` cleared.

    Imported lazily so the rest of this module (and the tests that only
    exercise the permutation floor) never pay shap's numba/llvmlite import
    cost. Raises on failure -- the caller isolates and degrades.
    """
    import shap

    if method == "shap_tree":
        return np.asarray(shap.TreeExplainer(model).shap_values(X))

    background = shap.maskers.Independent(X, max_samples=min(100, X.shape[0]))
    return np.asarray(shap.LinearExplainer(model, background, seed=seed).shap_values(X))


def _direction_for(signed_column: np.ndarray | None) -> str | None:
    """"increases"/"decreases" only when the feature pushes predictions
    consistently one way; "mixed" when the sign flips substantially across
    rows (a real property of interaction-heavy tree models, and honest to
    report as such rather than averaging it away).
    """
    if signed_column is None:
        return None
    nonzero = signed_column[signed_column != 0]
    if nonzero.size == 0:
        return None
    positive_share = float((nonzero > 0).mean())
    if positive_share >= _DIRECTION_MIXED_RATIO:
        return "increases"
    if positive_share <= 1.0 - _DIRECTION_MIXED_RATIO:
        return "decreases"
    return "mixed"


def build_attributions(
    feature_names: list[str],
    perm_mean: np.ndarray,
    perm_std: np.ndarray,
    shap_values: np.ndarray | None,
    top_k: int,
) -> tuple[list[FeatureAttribution], dict[str, float], list[str]]:
    """Assemble per-feature attributions, the source-column rollup, and the
    unused-feature list.

    The rollup and unused list are computed over *all* features; only the
    detailed `top_features` list is truncated to top_k, so a wide feature
    matrix stays bounded in LLM context without the column-level summary
    silently losing mass.
    """
    n_features = len(feature_names)
    mean_abs = _mean_abs_shap(shap_values, n_features) if shap_values is not None else None
    signed = _normalize_shap_values(shap_values, n_features) if shap_values is not None else None
    if signed is not None and signed.shape[1] != n_features:
        signed = None

    # Rollup uses permutation importance clipped at 0: a negative value means
    # shuffling the feature *improved* the score, i.e. noise, and letting that
    # subtract from a sibling one-hot column's contribution would understate
    # the source column.
    column_totals: dict[str, float] = {}
    for name, imp in zip(feature_names, perm_mean):
        column_totals[source_column_of(name)] = column_totals.get(
            source_column_of(name), 0.0
        ) + max(float(imp), 0.0)
    total = sum(column_totals.values())
    column_importance = (
        {k: round(v / total, 4) for k, v in sorted(column_totals.items(), key=lambda p: -p[1])}
        if total > 0
        else {k: 0.0 for k in column_totals}
    )

    unused = [name for name, imp in zip(feature_names, perm_mean) if float(imp) <= 0.0]

    order = np.argsort(-np.asarray(perm_mean))[:top_k]
    attributions = [
        FeatureAttribution(
            feature=feature_names[i],
            source_column=source_column_of(feature_names[i]),
            permutation_importance=round(float(perm_mean[i]), 6),
            permutation_std=round(float(perm_std[i]), 6),
            mean_abs_shap=round(float(mean_abs[i]), 6) if mean_abs is not None else None,
            direction=_direction_for(signed[:, i] if signed is not None else None),
        )
        for i in order
    ]
    return attributions, column_importance, unused


def _pick_example_rows(
    model: Any, X: np.ndarray, y: pd.Series, task_type: TaskType, k: int
) -> list[tuple[str, int, float | None]]:
    """Choose (kind, row_position, proba) triples worth explaining in detail.

    For classification: most-confident-correct, most-confident-*wrong* (the
    single most useful row for a reviewer -- it shows what the model believes
    when it is badly mistaken), and the most-uncertain rows. For regression
    there are no probabilities, so buckets fall back to residual magnitude.
    """
    preds = model.predict(X)
    y_arr = np.asarray(y)
    picks: list[tuple[str, int, float | None]] = []

    if task_type == "classification":
        proba_fn = getattr(model, "predict_proba", None)
        if proba_fn is None:
            return picks
        try:
            confidence = np.max(np.asarray(proba_fn(X)), axis=1)
        except Exception:  # noqa: BLE001 -- local examples are optional detail, never fatal
            return picks
        correct = preds == y_arr
        for kind, mask, key in (
            ("confident_correct", correct, -confidence),
            ("confident_wrong", ~correct, -confidence),
            ("most_uncertain", np.ones_like(correct, dtype=bool), np.abs(confidence - 0.5)),
        ):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            for i in idx[np.argsort(key[idx])][:k]:
                picks.append((kind, int(i), round(float(confidence[i]), 4)))
        return picks

    residuals = np.abs(np.asarray(preds, dtype=float) - y_arr.astype(float))
    order = np.argsort(residuals)
    for kind, idx in (
        ("confident_correct", order[:k]),
        ("confident_wrong", order[::-1][:k]),
    ):
        picks.extend((kind, int(i), None) for i in idx)
    return picks


def build_local_explanations(
    model: Any,
    X: np.ndarray,
    y: pd.Series,
    feature_names: list[str],
    signed_shap: np.ndarray,
    task_type: TaskType,
    k: int,
    top_k: int,
) -> list[LocalExplanation]:
    preds = model.predict(X)
    y_arr = np.asarray(y)
    out: list[LocalExplanation] = []
    for kind, pos, proba in _pick_example_rows(model, X, y, task_type, k):
        row = signed_shap[pos]
        top = np.argsort(-np.abs(row))[:top_k]
        out.append(
            LocalExplanation(
                kind=kind,
                row_position=pos,
                actual=str(y_arr[pos]),
                predicted=str(preds[pos]),
                predicted_proba=proba,
                top_contributions={
                    feature_names[i]: round(float(row[i]), 6) for i in top if row[i] != 0
                },
            )
        )
    return out


def fit_surrogate(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    task_type: TaskType,
    max_depth: int,
    seed: int,
) -> tuple[str, float]:
    """A shallow tree fit to the *model's own predictions* (not the true
    labels), so its rules describe the model's decision surface rather than
    the data. `fidelity` is how well it reproduces the model -- low fidelity
    is itself the finding: it means no simple rule set captures this model,
    and the rules should not be trusted as an explanation.
    """
    model_preds = model.predict(X)
    cls = DecisionTreeClassifier if task_type == "classification" else DecisionTreeRegressor
    surrogate = cls(max_depth=max_depth, random_state=seed).fit(X, model_preds)
    surrogate_preds = surrogate.predict(X)
    fidelity = (
        accuracy_score(model_preds, surrogate_preds)
        if task_type == "classification"
        else r2_score(model_preds, surrogate_preds)
    )
    return export_text(surrogate, feature_names=list(feature_names)), round(float(fidelity), 4)


def explain_model(
    model: Any,
    model_name: str,
    X: np.ndarray,
    y: pd.Series,
    y_train: pd.Series,
    feature_names: list[str],
    task_type: TaskType,
    metric: str,
    run_id: str = "",
) -> ExplanationReport:
    """Full explanation for one already-evaluated model. Never raises for a
    partial failure -- SHAP, local examples, and the surrogate each degrade
    independently into `warnings`, because a report missing its SHAP layer is
    still useful to a reviewer while a raised exception loses everything.
    """
    warnings: list[str] = []
    seed = settings.RANDOM_SEED

    X_sample, y_sample, _ = _subsample(X, y, settings.EXPLAIN_MAX_ROWS, seed)

    perm_mean, perm_std = compute_permutation_importance(
        model, X_sample, y_sample, metric, y_train, settings.EXPLAIN_PERMUTATION_REPEATS, seed
    )

    n_classes = int(np.unique(y_train).size) if task_type == "classification" else None
    shap_values: np.ndarray | None = None
    method: AttributionMethod = "permutation_only"
    planned = shap_method_for(model_name, n_classes)
    if planned is None and model_name in _SHAP_METHOD_BY_MODEL:
        warnings.append(
            f"SHAP skipped for '{model_name}' on a {n_classes}-class target: shap's "
            "TreeExplainer segfaults on multiclass CatBoost, which would kill the run. "
            "Reporting permutation importance only."
        )
    elif planned is not None:
        try:
            shap_values = compute_shap_values(model, planned, X_sample, seed)
            method = planned
        except Exception as exc:  # noqa: BLE001 -- degrade to the permutation floor, never fail
            warnings.append(
                f"SHAP unavailable for '{model_name}' ({type(exc).__name__}: {exc}); "
                "reporting permutation importance only."
            )
            shap_values, method = None, "permutation_only"

    top_features, column_importance, unused = build_attributions(
        feature_names, perm_mean, perm_std, shap_values, settings.EXPLAIN_TOP_K_FEATURES
    )

    local: list[LocalExplanation] = []
    if shap_values is not None:
        try:
            signed = _normalize_shap_values(shap_values, len(feature_names))
            if signed.shape[1] == len(feature_names):
                local = build_local_explanations(
                    model,
                    X_sample,
                    y_sample,
                    feature_names,
                    signed,
                    task_type,
                    settings.EXPLAIN_LOCAL_EXAMPLES,
                    settings.EXPLAIN_TOP_K_FEATURES,
                )
        except Exception as exc:  # noqa: BLE001 -- optional detail, not worth failing the report
            warnings.append(f"Local explanations skipped ({type(exc).__name__}: {exc}).")

    surrogate_rules: str | None = None
    surrogate_fidelity: float | None = None
    try:
        surrogate_rules, surrogate_fidelity = fit_surrogate(
            model,
            X_sample,
            feature_names,
            task_type,
            settings.EXPLAIN_SURROGATE_MAX_DEPTH,
            seed,
        )
    except Exception as exc:  # noqa: BLE001 -- optional detail
        warnings.append(f"Surrogate model skipped ({type(exc).__name__}: {exc}).")

    top_column = next(iter(column_importance), None)
    notes = (
        f"Top source column '{top_column}' carries "
        f"{column_importance.get(top_column, 0.0):.1%} of measured importance."
        if top_column
        else "No feature showed measurable permutation importance."
    )
    if unused:
        notes += f" {len(unused)} of {len(feature_names)} engineered features contributed nothing."

    return ExplanationReport(
        run_id=run_id,
        model_name=model_name,
        metric_name=metric,
        explained_on="test",
        n_rows_explained=int(X_sample.shape[0]),
        attribution_method=method,
        top_features=top_features,
        column_importance=column_importance,
        local_explanations=local,
        surrogate_rules=surrogate_rules,
        surrogate_fidelity=surrogate_fidelity,
        unused_features=unused,
        warnings=warnings,
        notes=notes,
    )


class ExplainModelsInput(BaseModel):
    model_names: list[str] = Field(
        default_factory=list,
        description="Evaluated models to explain. Empty means every evaluated model.",
    )


class ExplainModelsTool(Tool):
    name: str = "explain_models"
    description: str = (
        "Explains what an already-evaluated model learned: permutation importance for every "
        "model (including the ensemble, which has no feature_importances_ of its own), SHAP "
        "attributions where exact, importance rolled up to original dataset columns, example "
        "rows including the model's most confident mistakes, and a surrogate decision tree "
        "with a fidelity score. Read-only. Requires evaluate_models to have run first; call "
        "EXACTLY ONCE. model_names defaults to every evaluated model."
    )
    args_schema: type[BaseModel] = ExplainModelsInput
    run_id: str = ""

    def _run(self, model_names: list[str] | None = None) -> str:
        state = get_data_store().get(self.run_id)

        if state.explanation_applied:
            return json.dumps(
                {
                    "error": "explain_models has already been called for this run. Do not call "
                    "it again -- proceed to the next task with the existing explanations."
                }
            )
        # Ordering guard, not just a convenience check: this is what makes
        # "X_test is scored exactly once" enforceable rather than aspirational.
        # Explanation may only ever read X_test *after* scoring is locked in,
        # so nothing it reveals can be fed back into choosing a model.
        if not state.evaluation_applied:
            return json.dumps(
                {"error": "Run evaluate_models first; models must be scored before they can be explained."}
            )
        if not state.evaluation_reports:
            return json.dumps({"error": "No evaluation reports found for this run."})

        requested = list(model_names) if model_names else sorted(state.evaluation_reports)
        invalid = sorted(set(requested) - set(state.evaluation_reports))
        if invalid:
            return json.dumps(
                {
                    "error": f"{invalid} have no evaluation report; explainable models are "
                    f"{sorted(state.evaluation_reports)}."
                }
            )

        # Locked in before X_test is read, matching EvaluateModelsTool's
        # placement: a call that fails partway through still consumed the
        # resource, so the flag must not depend on a clean finish.
        state.explanation_applied = True

        metric = state.leaderboard.metric_name
        reports = []
        for name in requested:
            report = explain_model(
                state.fitted_models[name],
                name,
                state.X_test,
                state.y_test,
                state.y_train,
                state.feature_names,
                state.task_type,
                metric,
                run_id=self.run_id,
            )
            state.explanation_reports[name] = report
            reports.append(report)
            log_json_artifact(
                state.mlflow_run_id, report, f"explanation/{name}_report.json"
            )
            if report.surrogate_fidelity is not None:
                log_stage_metrics(
                    state.mlflow_run_id,
                    "explanation",
                    {f"{name}_surrogate_fidelity": report.surrogate_fidelity},
                )

        state.record("explanation", "explained", {"models": requested})
        return ExplanationBundle(run_id=self.run_id, reports=reports).model_dump_json()
